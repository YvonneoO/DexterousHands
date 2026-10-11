"""Startup watchdog for the online Pred-Tac sim processes (stdlib only).

IsaacGym's JIT extension load ("Loading extension module gymtorch...") can hang forever on a bad node (seen on dgx069, dgx072): the job then
sits until its walltime and, in a chain, so does every later link. This turns the hang into a quick, loud failure:

  start()    first thing in the entry script (before the isaacgym imports). Active only for online runs (PREDTAC_RUN_ID set, or
             STARTUP_WATCHDOG=1) inside a SLURM job. Arms faulthandler.dump_traceback_later(timeout, exit=True): if progress() is not called
             in time, the Python stack of the hung process is dumped to stderr and the process exits with status 1 -- faulthandler runs in its own
             C thread, so it works even while the main thread is stuck inside dlopen / CUDA init holding the GIL.
             It also launches a detached bash watcher: when the marker file does not appear in time it adds this host to ExcNodeList of all of
             this user's PENDING jobs (so the following chain links avoid the node) and scancels this job.
  progress() called from the training loop (once per iteration): the first call ends the startup phase; from then on every call re-arms a
             STALL timer (faulthandler, exit=True) and refreshes the marker's mtime. If no iteration completes for STALL_WATCHDOG_S the process
             dumps its stack and exits 1, and the bash watcher scancels the job (no node exclusion: a stalled server is as likely as a bad
             node, and a bad node is caught by the startup watchdog of the next link anyway).

Knobs: STARTUP_WATCHDOG_S (default 1200 s, counted from start()), STALL_WATCHDOG_S (default 900 s between two iterations; the synchronous
mode waits up to PREDTAC_BLOCK_TIMEOUT_S per capture, so a dead server stretches one iteration to many minutes), STARTUP_WATCHDOG=0 disables.
"""
import faulthandler
import os
import subprocess
import sys

_S = {"armed": False, "marker": None}

_WATCHER = r'''
JOB="$1"; MARK="$2"; T="$3"; LOG="$4"; STALL="$5"; PARENT="$6"; HOST="$(hostname)"
end=$((SECONDS + T + 45))
while [ "$SECONDS" -lt "$end" ]; do [ -f "$MARK" ] && break; sleep 5; done
if [ -f "$MARK" ]; then
  # training started: from here on watch the marker's age (refreshed by every PPO iteration)
  while kill -0 "$PARENT" 2>/dev/null; do
    sleep 30
    now="$(date +%s)"; mt="$(stat -c %Y "$MARK" 2>/dev/null || echo "$now")"
    if [ $((now - mt)) -gt "$STALL" ]; then
      echo "$(date -Is) stall watchdog: job $JOB on $HOST: no PPO iteration for $((now - mt))s -> cancelling the job" >> "$LOG"
      scancel "$JOB"; exit 0
    fi
  done
  exit 0
fi
echo "$(date -Is) startup watchdog: job $JOB on $HOST made no training progress in ${T}s -> excluding the node, cancelling the job" >> "$LOG"
for j in $(squeue -u "$(id -un)" -h -t PD -o %i 2>/dev/null); do
  cur="$(scontrol show job "$j" 2>/dev/null | tr ' ' '\n' | sed -n 's/^ExcNodeList=//p')"
  [ "$cur" = "(null)" ] && cur=""
  case ",$cur," in *",$HOST,"*) ;; *) scontrol update JobId="$j" ExcNodeList="${cur:+$cur,}$HOST" 2>/dev/null ;; esac
done
scancel "$JOB"
'''


def start(timeout_s=None):
    if _S["armed"] or os.environ.get("STARTUP_WATCHDOG", "") == "0":
        return
    job = os.environ.get("SLURM_JOB_ID")
    if not job or not (os.environ.get("PREDTAC_RUN_ID") or os.environ.get("STARTUP_WATCHDOG") == "1"):
        return
    t = int(timeout_s or os.environ.get("STARTUP_WATCHDOG_S", "1200"))
    root = os.environ.get("PREDTAC_IPC_ROOT") or os.getcwd()
    os.makedirs(root, exist_ok=True)
    marker = os.path.join(root, "watchdog_%s.alive" % job)
    if os.path.exists(marker):
        os.remove(marker)
    _S["marker"] = marker
    faulthandler.dump_traceback_later(t, exit=True, file=sys.stderr)
    try:
        subprocess.Popen(["bash", "-c", _WATCHER, "watcher", job, marker, str(t), os.path.join(root, "bad_nodes.log"),
                          str(int(os.environ.get("STALL_WATCHDOG_S", "900"))), str(os.getpid())],
                         start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass
    _S["armed"] = True
    print("[watchdog] startup watchdog armed: %d s to reach the first training iteration (job %s)" % (t, job), flush=True)


def progress():
    if not _S["armed"]:
        return
    stall = int(os.environ.get("STALL_WATCHDOG_S", "900"))
    faulthandler.dump_traceback_later(stall, exit=True, file=sys.stderr)      # re-arms (replaces the previous timer)
    try:
        with open(_S["marker"], "a"):
            pass
        os.utime(_S["marker"], None)
    except OSError:
        pass
    if not _S.get("seen"):
        _S["seen"] = True
        print("[watchdog] training progress seen: startup watchdog done, stall watchdog armed (%d s between iterations)" % stall, flush=True)
