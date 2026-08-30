# Third-Party Code Notices

This file records third-party source adapted into `pyimgano`. Local changes do
not remove the upstream copyright or license terms.

## PyOD

- Upstream: [yzhao062/pyod](https://github.com/yzhao062/pyod) @
  `34f7996effac700a5166d882d5e94c6e6078fae3`
- License: BSD-2-Clause
- Copyright: Copyright (c) 2018, Yue Zhao. All rights reserved.
- Adapted files:
  - `pyimgano/models/abod.py`
  - `pyimgano/models/cof.py`
  - `pyimgano/models/hbos.py`
  - `pyimgano/models/inne.py`
  - `pyimgano/models/kpca.py`
  - `pyimgano/models/lmdd.py`
  - `pyimgano/models/loci.py`
  - `pyimgano/models/mcd.py`
  - `pyimgano/models/ocsvm.py`
  - `pyimgano/models/qmcd.py`
- Notes: These implementations were adapted to the native PyImgAno estimator
  API and have subsequently diverged. PyOD is not a runtime dependency.

The following BSD-2-Clause terms apply to the adapted PyOD source:

> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> * Redistributions of source code must retain the above copyright notice,
>   this list of conditions and the following disclaimer.
>
> * Redistributions in binary form must reproduce the above copyright notice,
>   this list of conditions and the following disclaimer in the documentation
>   and/or other materials provided with the distribution.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
> ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
> LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
> CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
> SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
> INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
> CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
> ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
> POSSIBILITY OF SUCH DAMAGE.
