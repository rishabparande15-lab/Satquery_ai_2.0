# Send-ready clarification email — not sent

**Subject:** Clarification request: CDVQA–SECOND image mapping and temporal ordering

Dear CDVQA/SECOND authors,

I am working on SatQuery, an academic remote-sensing research project. We are
reviewing the official CDVQA annotations at revision
`cc5893123dd32326de38745b65d2ffe45055937b` for TRAIN and VALIDATION only;
we have not accessed any TEST or TEST2 annotations (`TEST access = 0`).

Could you please clarify the following?

1. How does CDVQA `file_name` map to physical SECOND image files?

2. Which SECOND image/folder is the earlier/pre-change image and which is the later/post-change image?

3. Are CDVQA TRAIN and VALIDATION identities directly recoverable from SECOND filenames?

4. What usage/license terms apply to SECOND imagery?

5. Is there a TRAIN/VALIDATION-specific file list or manifest that avoids reserved CDVQA TEST identities?

6. Are semantic maps/change masks available for those same identities?

For context, example admitted TRAIN stems are `00003.png`, `00011.png`, and
`00013.png`; example admitted VALIDATION stems are `00020.png`, `00060.png`,
and `00079.png`. If available, an official loader, filename mapping script,
README, split manifest, or archive file tree would be very helpful.

Thank you for your time and for making this work available.

Kind regards,

[Your name]
