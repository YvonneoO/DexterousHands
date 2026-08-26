#!/usr/bin/env python3
"""Project Isaac Gym rigid contacts onto the canonical EgoTouch 21x21 hand grid.

The mapping is virtual: it does not alter ShadowHand collision geometry or policy dynamics.
Each normal contact force is distributed to nearby taxels with normalized Gaussian weights,
then converted to pressure using the represented physical surface area of every taxel:

    pressure [Pa] = allocated normal force [N] / taxel area [m^2]

The Gaussian weights sum to one, so total normal force is conserved per contact.
"""

import json
import math
import os

import numpy as np


GRID_SIZE = 21


def _load_valid_cells(mapping_path):
    with open(mapping_path) as handle:
        mapping = json.load(handle)
    cells = {tuple(int(v) for v in key.split(",")) for key in mapping}
    if len(cells) != 217:
        raise ValueError("Expected 217 EgoTouch cells in {}, found {}".format(mapping_path, len(cells)))
    return cells


def _layout(side, valid):
    """Return canonical anatomical cell groups; counts are EgoTouch's 48/60/30/79."""
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")

    # EgoTouch uses the same canonical chart mask for both sides (the two mapping JSONs
    # have identical row/column keys). Handedness is represented by mapping values, not
    # by horizontally mirroring the 21x21 output chart.
    finger_cols = {"ff": 1, "mf": 5, "rf": 9, "lf": 13}
    thumb_cols = range(17, 20)

    groups = {}
    fingertip = set()
    finger_middle = set()
    for finger, col0 in finger_cols.items():
        groups[finger + "distal"] = {(row, col) for row in range(2, 6) for col in range(col0, col0 + 3)}
        groups[finger + "middle"] = {(row, col) for row in range(7, 10) for col in range(col0, col0 + 3)}
        groups[finger + "proximal"] = {(row, col) for row in range(10, 12) for col in range(col0, col0 + 3)}
        fingertip |= groups[finger + "distal"]
        finger_middle |= groups[finger + "middle"] | groups[finger + "proximal"]

    groups["thdistal"] = {(row, col) for row in range(6, 10) for col in thumb_cols}
    groups["thmiddle"] = {(row, col) for row in (11, 12) for col in thumb_cols}
    groups["thproximal"] = {(row, col) for row in range(14, 18) for col in thumb_cols}
    thumb = groups["thdistal"] | groups["thmiddle"] | groups["thproximal"]

    groups["palm"] = {
        (row, col)
        for row in range(13, 18)
        for col in range(1, 20)
        if col not in thumb_cols
    }
    little_col0 = finger_cols["lf"]
    groups["lfmetacarpal"] = {
        (row, col) for row in (13, 14) for col in range(little_col0, little_col0 + 3)
    }

    for name in groups:
        groups[name] &= valid

    covered = fingertip | finger_middle | thumb | groups["palm"]
    if (len(fingertip), len(finger_middle), len(thumb), len(groups["palm"])) != (48, 60, 30, 79):
        raise AssertionError("Unexpected EgoTouch anatomical counts")
    if covered != valid:
        raise AssertionError("Anatomical regions do not exactly cover the EgoTouch mask")
    return groups


def _centers(cells, x_min, x_max, z_min, z_max):
    """Assign grid cells evenly to local body x/z coordinates."""
    ordered = sorted(cells)
    rows = sorted({rc[0] for rc in ordered})
    cols = sorted({rc[1] for rc in ordered})
    row_to_z = {
        row: z_max - (z_max - z_min) * ((i + 0.5) / max(1, len(rows)))
        for i, row in enumerate(rows)
    }
    col_to_x = {
        col: x_min + (x_max - x_min) * ((i + 0.5) / max(1, len(cols)))
        for i, col in enumerate(cols)
    }
    rc = np.asarray(ordered, dtype=np.int32)
    xz = np.asarray([[col_to_x[col], row_to_z[row]] for row, col in ordered], dtype=np.float64)
    pitch_x = max((x_max - x_min) / max(1, len(cols)), 1e-5)
    pitch_z = max((z_max - z_min) / max(1, len(rows)), 1e-5)
    return rc, xz, pitch_x, pitch_z


def _body_geometry(suffix):
    """Projected palmar pad width/length in metres from the ShadowHand collision geometry."""
    if suffix.endswith("distal") and not suffix.startswith("th"):
        return 0.01410, 0.0260
    if suffix.endswith("middle") and not suffix.startswith("th"):
        return 0.01610, 0.0250
    if suffix.endswith("proximal") and not suffix.startswith("th"):
        return 0.02000, 0.0450
    if suffix == "thdistal":
        return 0.01836, 0.0275
    if suffix == "thmiddle":
        return 0.02200, 0.0320
    if suffix == "thproximal":
        return 0.02600, 0.0380
    if suffix == "lfmetacarpal":
        return 0.01900, 0.0500
    raise KeyError(suffix)


class EgoTouchTaxelMapper:
    """Map object contacts on one ShadowHand actor into an EgoTouch pressure grid."""

    def __init__(self, gym, env, actor_name, side, mapping_path):
        from isaacgym import gymapi

        self.gym = gym
        self.env = env
        self.side = side
        self.actor_name = actor_name
        self.mapping_path = os.path.abspath(mapping_path)
        self.valid_cells = _load_valid_cells(mapping_path)
        self.groups = _layout(side, self.valid_cells)
        self.valid_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        for row, col in self.valid_cells:
            self.valid_mask[row, col] = True

        actor = gym.find_actor_handle(env, actor_name)
        if actor < 0:
            raise RuntimeError("Cannot find actor {}".format(actor_name))
        object_actor = gym.find_actor_handle(env, "object")
        if object_actor < 0:
            raise RuntimeError("Cannot find object actor")
        # Articulated tasks (BottleCap, doors, scissors, etc.) have several
        # rigid bodies under the task's `object` actor.  Restricting tactile
        # projection to local body 0 silently dropped contacts on the moving
        # cap/link—the very contacts that constitute a successful rollout.
        object_body_count = gym.get_actor_rigid_body_count(env, object_actor)
        self.object_bodies = {
            int(gym.get_actor_rigid_body_index(env, object_actor, local_index, gymapi.DOMAIN_ENV))
            for local_index in range(object_body_count)
        }

        names = gym.get_actor_rigid_body_names(env, actor)
        self.hand_bodies = {
            int(gym.get_actor_rigid_body_index(env, actor, local_index, gymapi.DOMAIN_ENV)): full_name
            for local_index, full_name in enumerate(names)
        }
        self.body_specs = {}
        self.taxel_area_m2 = np.full((GRID_SIZE, GRID_SIZE), np.nan, dtype=np.float64)

        # Palm area is the union of the two collision-box palmar faces.
        palm_area = (0.064 * 0.098) + (0.022 * 0.050)
        palm_cells = self.groups["palm"]
        for row, col in palm_cells:
            self.taxel_area_m2[row, col] = palm_area / len(palm_cells)

        for local_index, full_name in enumerate(names):
            suffix = full_name.split(":")[-1]
            if suffix not in self.groups:
                continue
            cells = self.groups[suffix]
            if not cells:
                continue
            env_index = gym.get_actor_rigid_body_index(env, actor, local_index, gymapi.DOMAIN_ENV)
            if suffix == "palm":
                rc, xz, pitch_x, pitch_z = _centers(cells, -0.043, 0.043, -0.011, 0.087)
            else:
                width, length = _body_geometry(suffix)
                rc, xz, pitch_x, pitch_z = _centers(cells, -0.5 * width, 0.5 * width, 0.0, length)
                area = width * length / len(cells)
                for row, col in cells:
                    if not np.isfinite(self.taxel_area_m2[row, col]):
                        self.taxel_area_m2[row, col] = area
            self.body_specs[int(env_index)] = {
                "name": full_name,
                "suffix": suffix,
                "rc": rc,
                "xz": xz,
                "pitch_x": pitch_x,
                "pitch_z": pitch_z,
            }

        if not self.body_specs:
            raise RuntimeError("No tactile ShadowHand bodies found for {}".format(actor_name))
        if np.any(~np.isfinite(self.taxel_area_m2[self.valid_mask])):
            missing = np.argwhere(self.valid_mask & ~np.isfinite(self.taxel_area_m2))
            raise RuntimeError("Missing physical area for taxels: {}".format(missing.tolist()))

    @staticmethod
    def _vec3(value):
        if all(hasattr(value, axis) for axis in ("x", "y", "z")):
            return np.asarray([float(value.x), float(value.y), float(value.z)], dtype=np.float64)
        if isinstance(value, np.void) and value.dtype.names:
            return np.asarray([float(value[axis]) for axis in ("x", "y", "z")], dtype=np.float64)
        return np.asarray(value, dtype=np.float64).reshape(-1)[:3]

    @staticmethod
    def _field(contact, name):
        if isinstance(contact, np.void) and contact.dtype.names:
            aliases = {"local_pos0": "localPos0", "local_pos1": "localPos1"}
            return contact[aliases.get(name, name)]
        return getattr(contact, name)

    def project(self, contacts):
        force_grid_n = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
        source_force_n = 0.0
        contact_count = 0
        per_body_force = {}
        total_hand_object_contact_count = 0
        total_hand_object_force_n = 0.0
        unmapped_contact_count = 0
        unmapped_force_n = 0.0
        unmapped_body_force_n = {}
        unmapped_body_contact_count = {}

        for contact in contacts:
            body0 = int(self._field(contact, "body0"))
            body1 = int(self._field(contact, "body1"))
            if body0 in self.hand_bodies and body1 in self.object_bodies:
                hand_body = body0
                local = self._vec3(self._field(contact, "local_pos0"))
            elif body1 in self.hand_bodies and body0 in self.object_bodies:
                hand_body = body1
                local = self._vec3(self._field(contact, "local_pos1"))
            else:
                continue

            normal_force_n = max(0.0, float(self._field(contact, "lambda")))
            if not math.isfinite(normal_force_n) or normal_force_n <= 0.0:
                continue
            total_hand_object_contact_count += 1
            total_hand_object_force_n += normal_force_n
            if hand_body not in self.body_specs:
                unmapped_contact_count += 1
                unmapped_force_n += normal_force_n
                name = self.hand_bodies[hand_body]
                unmapped_body_force_n[name] = unmapped_body_force_n.get(name, 0.0) + normal_force_n
                unmapped_body_contact_count[name] = unmapped_body_contact_count.get(name, 0) + 1
                continue
            spec = self.body_specs[hand_body]
            # ShadowHand capsule/box longitudinal axis is local z; local x spans the pad width.
            dx = (local[0] - spec["xz"][:, 0]) / spec["pitch_x"]
            dz = (local[2] - spec["xz"][:, 1]) / spec["pitch_z"]
            weights = np.exp(-0.5 * (dx * dx + dz * dz))
            weight_sum = float(weights.sum())
            if weight_sum <= 1e-20:
                weights[:] = 0.0
                weights[int(np.argmin(dx * dx + dz * dz))] = 1.0
            else:
                weights /= weight_sum
            allocated = normal_force_n * weights
            for (row, col), force in zip(spec["rc"], allocated):
                force_grid_n[row, col] += float(force)
            source_force_n += normal_force_n
            contact_count += 1
            name = spec["suffix"]
            per_body_force[name] = per_body_force.get(name, 0.0) + normal_force_n

        pressure_pa = np.full((GRID_SIZE, GRID_SIZE), np.nan, dtype=np.float32)
        pressure_pa[self.valid_mask] = (
            force_grid_n[self.valid_mask] / self.taxel_area_m2[self.valid_mask]
        ).astype(np.float32)
        reconstructed_force_n = float(force_grid_n.sum())
        return pressure_pa, force_grid_n.astype(np.float32), {
            "source_force_n": float(source_force_n),
            "reconstructed_force_n": reconstructed_force_n,
            "force_error_n": reconstructed_force_n - float(source_force_n),
            "contact_count": int(contact_count),
            "per_body_force_n": per_body_force,
            "coverage_available": True,
            "total_hand_object_contact_count": int(total_hand_object_contact_count),
            "mapped_hand_object_contact_count": int(contact_count),
            "unmapped_hand_object_contact_count": int(unmapped_contact_count),
            "total_hand_object_normal_force_n": float(total_hand_object_force_n),
            "mapped_hand_object_normal_force_n": float(source_force_n),
            "unmapped_hand_object_normal_force_n": float(unmapped_force_n),
            "mapped_force_fraction": (
                float(source_force_n) / float(total_hand_object_force_n)
                if total_hand_object_force_n > 0.0 else float("nan")
            ),
            "unmapped_body_force_n": unmapped_body_force_n,
            "unmapped_body_contact_count": unmapped_body_contact_count,
        }

    def project_net_forces(self, net_contact_forces, min_force_n=1.0e-5):
        """Project Isaac Gym net contact force tensor onto the EgoTouch grid.

        Isaac Gym's per-contact API is unavailable after GPU-pipeline simulation
        starts, but the net-contact-force tensor is supported.  The tensor is
        indexed by env rigid body id and contains the summed contact force vector
        for each body.  It does not contain contact pairs or local contact
        coordinates, so this method distributes each tactile body's net force
        uniformly over that body's canonical EgoTouch cells.
        """
        forces = np.asarray(net_contact_forces, dtype=np.float64)
        force_grid_n = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
        source_force_n = 0.0
        contact_count = 0
        per_body_force = {}

        for body_index, spec in self.body_specs.items():
            if body_index < 0 or body_index >= forces.shape[0]:
                continue
            force_n = float(np.linalg.norm(forces[body_index, :3]))
            if not math.isfinite(force_n) or force_n <= min_force_n:
                continue
            cells = spec["rc"]
            if len(cells) == 0:
                continue
            per_cell = force_n / float(len(cells))
            for row, col in cells:
                force_grid_n[row, col] += per_cell
            source_force_n += force_n
            contact_count += 1
            name = spec["suffix"]
            per_body_force[name] = per_body_force.get(name, 0.0) + force_n

        pressure_pa = np.full((GRID_SIZE, GRID_SIZE), np.nan, dtype=np.float32)
        pressure_pa[self.valid_mask] = (
            force_grid_n[self.valid_mask] / self.taxel_area_m2[self.valid_mask]
        ).astype(np.float32)
        reconstructed_force_n = float(force_grid_n.sum())
        return pressure_pa, force_grid_n.astype(np.float32), {
            "source_force_n": float(source_force_n),
            "reconstructed_force_n": reconstructed_force_n,
            "force_error_n": reconstructed_force_n - float(source_force_n),
            "contact_count": int(contact_count),
            "per_body_force_n": per_body_force,
            "projection": "net_contact_force_tensor_uniform_body_distribution",
            "coverage_available": False,
            "total_hand_object_contact_count": -1,
            "mapped_hand_object_contact_count": int(contact_count),
            "unmapped_hand_object_contact_count": -1,
            "total_hand_object_normal_force_n": float("nan"),
            "mapped_hand_object_normal_force_n": float(source_force_n),
            "unmapped_hand_object_normal_force_n": float("nan"),
            "mapped_force_fraction": float("nan"),
            "unmapped_body_force_n": {},
            "unmapped_body_contact_count": {},
        }

    def metadata(self):
        return {
            "side": self.side,
            "actor_name": self.actor_name,
            "valid_taxels": int(self.valid_mask.sum()),
            "object_body_env_indices": sorted(self.object_bodies),
            "all_hand_body_env_indices": {
                str(index): name for index, name in sorted(self.hand_bodies.items())
            },
            "body_env_indices": {
                str(index): spec["name"] for index, spec in sorted(self.body_specs.items())
            },
            "unmapped_hand_body_env_indices": {
                str(index): name
                for index, name in sorted(self.hand_bodies.items())
                if index not in self.body_specs
            },
            "pressure_unit": "Pa",
            "force_unit": "N",
            "area_unit": "m^2",
            "mapping_file": self.mapping_path,
        }
