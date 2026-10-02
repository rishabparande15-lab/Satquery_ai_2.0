# CDVQA and SECOND clarification request

## Questions

1. How does CDVQA `file_name` map to physical SECOND image files?

2. Which SECOND image/folder is the earlier/pre-change image and which is the later/post-change image?

3. Are CDVQA TRAIN and VALIDATION identities directly recoverable from SECOND filenames?

4. What usage/license terms apply to SECOND imagery?

5. Is there a TRAIN/VALIDATION-specific file list or manifest that avoids reserved CDVQA TEST identities?

6. Are semantic maps/change masks available for those same identities?

## Email draft

Subject: Clarification on CDVQA-to-SECOND image linkage and permitted use

Dear CDVQA/SECOND authors,

I am preparing a reproducible research-only implementation using the official
CDVQA annotations at commit `cc5893123dd32326de38745b65d2ffe45055937b`. The
TRAIN and VALIDATION annotations provide pair stems, but the publicly available
documentation does not establish their corresponding SECOND file paths or the
chronological meaning of the pair members.

To avoid any use of reserved test identities and to follow the intended terms,
I would appreciate clarification on the six points above. For context, example
admitted stems are included in the accompanying evidence bundle; no TEST or
TEST2 records are included.

Thank you for your time and for making this work available.

Kind regards,

[Your name]
