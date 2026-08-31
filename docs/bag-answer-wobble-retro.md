# BAG retroactive answer-wobble audit

Detector: pure structural — counts a path as **wobbled** when the agent's bash trace contains two or more recoverable writes to that path with **distinct content**. Opaque writes (cp/mv, pipes, variable interpolation, append `>>`, printf with `%`-format specifiers) are intentionally ignored.

## Per-task-family summary

| family | trials | w/trace | wins | wobbled | wobbled-wins | wobble-rate (of w/trace) |
|---|---:|---:|---:|---:|---:|---:|
| build-cython-ext | 36 | 25 | 22 | 1 | 1 | 4.0% |
| chess-best-move | 42 | 24 | 22 | 0 | 0 | 0.0% |
| configure-git-webserver | 36 | 25 | 22 | 8 | 7 | 32.0% |
| fix-code-vulnerability | 38 | 25 | 25 | 4 | 4 | 16.0% |
| instance_nodebb__nodebb-04998908 | 1 | 0 | 0 | 0 | 0 | 0.0% |
| log-summary-date-ranges | 36 | 25 | 32 | 0 | 0 | 0.0% |
| polyglot-c-py | 39 | 26 | 23 | 12 | 9 | 46.2% |
| qemu-alpine-ssh | 36 | 13 | 12 | 11 | 9 | 84.6% |
| qemu-startup | 38 | 9 | 18 | 6 | 4 | 66.7% |
| regex-log | 43 | 28 | 35 | 1 | 1 | 3.6% |
| sqlite-with-gcov | 38 | 23 | 21 | 0 | 0 | 0.0% |

**Aggregate:** 43/223 traced trials (19.3%) show wobble; 35/232 wins (15.1%) contain wobble — these are the trials a stricter verifier might flip.

## Chess-best-move drill-down

_No chess trials with structurally-recovered multi-write wobble._

NOTE: the deep-dive doc identified a different masquerade — a single heredoc that emits two newline-separated UCI moves. That pattern is **one write with two answers**, not two writes; the structural detector here intentionally does NOT flag single-write-multi-line content (we cannot tell `e2e4\ng2g4` apart from `final-move\nrationale`). It is captured by the verifier-strictness recommendation, not by this scan.

## Sample wobble events (up to 12)

- `polyglot-c-py` / `polyglot-c-py__XcUEHSH` -> `/app/polyglot/main.py.c` (reward=1.0)
    - cmd#2 digest=94248fdcf871 bytes=444 preview='#include <stdio.h>\\n#include <stdlib.h>\\n#if 0\\n"""\\n#endif\\n\\nint main(int argc'
    - cmd#4 digest=982adbb4b7a7 bytes=441 preview='#include <stdio.h>\\n#include <stdlib.h>\\n#define Q(x)\\nQ(""")\\n\\nint main(int ar'
    - cmd#5 digest=39636dab657d bytes=442 preview='#if 0\\n"""\\n#endif\\n#include <stdio.h>\\n#include <stdlib.h>\\nint main(int argc, '
    - cmd#7 digest=405278513ccc bytes=1015 preview='#if 0\\n"""\\n#endif\\n#include <stdio.h>\\n#include <stdlib.h>\\n#include <string.h>'
- `polyglot-c-py` / `polyglot-c-py__VpmDPy2` -> `/app/polyglot/main.py.c` (reward=1.0)
    - cmd#2 digest=341fd2ccbf07 bytes=381 preview='#if 0\\n"""\\n#endif\\n#include <stdio.h>\\n#include <stdlib.h>\\nint main(int argc, '
    - cmd#5 digest=27c0938781bd bytes=390 preview='#if 0\\n""" /*\\n#endif\\n#include <stdio.h>\\n#include <stdlib.h>\\nint main(int arg'
    - cmd#6 digest=341fd2ccbf07 bytes=381 preview='#if 0\\n"""\\n#endif\\n#include <stdio.h>\\n#include <stdlib.h>\\nint main(int argc, '
- `qemu-alpine-ssh` / `qemu-alpine-ssh__U7bwtVP` -> `/tmp/shim.c` (reward=0.0)
    - cmd#27 digest=d1793bd1518f bytes=1635 preview='#define _GNU_SOURCE\\n#include <sys/syscall.h>\\n#include <unistd.h>\\n#include <si'
    - cmd#29 digest=fdeff037c87a bytes=401 preview='#define _GNU_SOURCE\\n#include <sys/syscall.h>\\n#include <unistd.h>\\n#include <si'
- `polyglot-c-py` / `polyglot-c-py__gKZ2A3v` -> `/app/polyglot/main.py.c` (reward=1.0)
    - cmd#2 digest=ea2364bf19fd bytes=393 preview='#if 0\\n"""\\n#endif\\n#include <stdio.h>\\n#include <stdlib.h>\\nint main(int argc, '
    - cmd#3 digest=ea2364bf19fd bytes=393 preview='#if 0\\n"""\\n#endif\\n#include <stdio.h>\\n#include <stdlib.h>\\nint main(int argc, '
    - cmd#8 digest=a3e67ff8a385 bytes=468 preview='#if 0\\n"""\\n#endif\\n#include <stdio.h>\\n#include <stdlib.h>\\nint main(int argc, '
- `polyglot-c-py` / `polyglot-c-py__axEyKVD` -> `/app/polyglot/main.py.c` (reward=0.0)
    - cmd#2 digest=b55508b3ec34 bytes=360 preview='#include <stdio.h>\\n#include <stdlib.h>\\n/*\\n"""\\nimport sys\\nn = int(sys.argv[1'
    - cmd#3 digest=15146f6ac884 bytes=374 preview='#include <stdio.h>\\n#include <stdlib.h>\\n#define x """\\nx = 1\\n"""\\nimport sys\\n'
    - cmd#4 digest=1f954dc6fc18 bytes=368 preview='#if 0\\n"""\\n#endif\\n#include <stdio.h>\\n#include <stdlib.h>\\nint main(int argc, '
- `configure-git-webserver` / `configure-git-webserver__XWeAqQe` -> `/git/server/.git/hooks/post-receive` (reward=0.0)
    - cmd#20 digest=04474580adf4 bytes=108 preview='#!/bin/bash\\nunset GIT_DIR\\ncd /git/server && git --git-dir=/git/server/.git --w'
    - cmd#21 digest=80c5a82b1fde bytes=90 preview='#!/bin/bash\\nunset GIT_DIR\\ngit --git-dir=/git/server/.git --work-tree=/var/www '
- `polyglot-c-py` / `polyglot-c-py__QEskgk5` -> `/app/polyglot/main.py.c` (reward=0.0)
    - cmd#2 digest=b0e7dd30401f bytes=413 preview='#include <stdio.h>\\n#include <stdlib.h>\\n/*\\n"""\\n#*/\\n#include <stdio.h>\\n#incl'
    - cmd#3 digest=171158f14600 bytes=404 preview='#include <stdio.h>\\n#include <stdlib.h>\\n#if 0\\n"""\\n#endif\\nint main(int argc, '
    - cmd#4 digest=cd60156cec5e bytes=428 preview='#include <stdio.h>\\n#include <stdlib.h>\\n#if 0\\n"""\\n#endif\\n/*\\n"""\\n*/\\nint ma'
    - cmd#5 digest=ba68cdd6abba bytes=404 preview='#include <stdio.h>\\n#include <stdlib.h>\\n#if 0\\n"""\\n#endif\\nint main(int argc, '
    - cmd#6 digest=eafe2f68d150 bytes=410 preview='#include <stdio.h>\\n#include <stdlib.h>\\n#if 0\\n"""  "\\n#endif\\nint main(int arg'
    - cmd#7 digest=ba68cdd6abba bytes=404 preview='#include <stdio.h>\\n#include <stdlib.h>\\n#if 0\\n"""\\n#endif\\nint main(int argc, '
    - cmd#8 digest=171158f14600 bytes=404 preview='#include <stdio.h>\\n#include <stdlib.h>\\n#if 0\\n"""\\n#endif\\nint main(int argc, '
- `polyglot-c-py` / `polyglot-c-py__QEskgk5` -> `/tmp/test.py.c` (reward=0.0)
    - cmd#14 digest=f4112de27f64 bytes=414 preview='#define PYCOMMENT 0\\n#if PYCOMMENT\\n"""\\n#endif\\n/* this part is C */\\n#include '
    - cmd#15 digest=662c5d6ae66a bytes=361 preview='#if 0\\n"""\\n#endif\\n#include <stdio.h>\\n#include <stdlib.h>\\nint main(int argc, '
    - cmd#16 digest=b88a95986768 bytes=378 preview='#if 0\\n"""\\n"""\\n#endif\\n#include <stdio.h>\\n#include <stdlib.h>\\nint main(int a'
    - cmd#18 digest=20e29337b98c bytes=424 preview='#include <stdio.h>\\n#include <stdlib.h>\\n#if 0\\n""" "\\n"""\\n" "\\n#endif\\nint mai'
- `configure-git-webserver` / `configure-git-webserver__qnDyJCK` -> `/git/server/hooks/post-receive` (reward=1.0)
    - cmd#6 digest=8ea5867422b1 bytes=51 preview='#!/bin/bash\\nGIT_WORK_TREE=/var/www git checkout -f\\n'
    - cmd#13 digest=ee34ab59ff53 bytes=73 preview='#!/bin/bash\\nGIT_WORK_TREE=/var/www git --git-dir=/git/server checkout -f\\n'
    - cmd#15 digest=ee34ab59ff53 bytes=73 preview='#!/bin/bash\\nGIT_WORK_TREE=/var/www git --git-dir=/git/server checkout -f\\n'
    - cmd#26 digest=ee34ab59ff53 bytes=73 preview='#!/bin/bash\\nGIT_WORK_TREE=/var/www git --git-dir=/git/server checkout -f\\n'
    - cmd#28 digest=ee34ab59ff53 bytes=73 preview='#!/bin/bash\\nGIT_WORK_TREE=/var/www git --git-dir=/git/server checkout -f\\n'
    - cmd#32 digest=07479db99cef bytes=281 preview='#!/bin/bash\\nexec >>/tmp/hook.log 2>&1\\necho "=== $(date) hook running as $(whoa'
    - cmd#39 digest=ee34ab59ff53 bytes=73 preview='#!/bin/bash\\nGIT_WORK_TREE=/var/www git --git-dir=/git/server checkout -f\\n'
    - cmd#45 digest=ee34ab59ff53 bytes=73 preview='#!/bin/bash\\nGIT_WORK_TREE=/var/www git --git-dir=/git/server checkout -f\\n'
- `fix-code-vulnerability` / `fix-code-vulnerability__KKWgie4` -> `report.jsonl` (reward=1.0)
    - cmd#15 digest=d55bd972fa45 bytes=47 preview='{"file_path": "bottle.py", "cwe_id": "CWE-93"}\\n'
    - cmd#19 digest=ebfb63dd13ab bytes=54 preview='{"file_path": "/app/bottle.py", "cwe_id": ["cwe-93"]}\\n'
- `qemu-startup` / `qemu-startup__iNU5PKS` -> `/tmp/nomq.c` (reward=0.0)
    - cmd#26 digest=8c6e9984d8e6 bytes=783 preview='#define _GNU_SOURCE\\n#include <fcntl.h>\\n#include <errno.h>\\n#include <stddef.h>'
    - cmd#28 digest=0184b4bb3609 bytes=285 preview='#define _GNU_SOURCE\\n#include <errno.h>\\n#include <stddef.h>\\n\\ntypedef int mqd_'
- `qemu-startup` / `qemu-startup__u4ydTLU` -> `/tmp/stub3.c` (reward=1.0)
    - cmd#44 digest=d7397f92a1e2 bytes=1153 preview='#define _GNU_SOURCE\\n#include <stdarg.h>\\n#include <stddef.h>\\n#include <stdio.h'
    - cmd#45 digest=e04c08a0cfec bytes=1244 preview='#define _GNU_SOURCE\\n#include <stdarg.h>\\n#include <stddef.h>\\n#include <stdio.h'

## Recommendation

35 winning trials contain structurally detectable multi-write wobble (15.1% of wins). That is high enough to call wobble a **signal worth acting on**:

1. Feed the `[Wobble scan]` block into the pre-submit self-check (already done in this PR). The auditor LLM will quote the conflicting versions and force another iteration.
2. Add a system-prompt rule: "once you have written a deliverable file, do not overwrite it unless you have new evidence the previous content was wrong; if you do overwrite, explain in chat what changed." This frames the rule as epistemic discipline rather than "commit to one answer".
3. Strengthen the verifier where ambiguity hides bugs (chess: first-line-only).
