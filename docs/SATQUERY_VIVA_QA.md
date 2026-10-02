# SatQuery viva and judge question bank

1. **What is SatQuery?**
   SatQuery is a local agentic assistant that routes remote-sensing questions to a specialist based on declared inputs and query intent.

2. **Why use an agentic architecture?**
   A single image, an S1/S2 pair, and a PRE/POST pair require different validation and model contracts. The agent keeps those routes explicit.

3. **Why not use one large VLM for every task?**
   One generic route can conceal modality and chronology mistakes. SatQuery uses narrower specialists and surfaces their provenance and limits.

4. **What is CROMA?**
   CROMA is the pretrained representation model used in the approved optical-SAR route to form joint S1/S2 representations.

5. **Why combine Sentinel-1 and Sentinel-2?**
   Sentinel-1 contributes SAR information and Sentinel-2 contributes multispectral optical information. The route requires both as a matched pair.

6. **What is SAR?**
   Synthetic Aperture Radar measures backscatter with active microwave sensing and can complement optical imagery under different observing conditions.

7. **Why does SatQuery use Qwen?**
   Frozen Qwen vision-language components generate language after the route-specific image representation is prepared.

8. **What is Chg2Cap?**
   Chg2Cap is the specialist used for PRE/POST change-caption generation in the temporal route.

9. **What are T1 and T2?**
   T1 is the earlier PRE image and T2 is the later POST image. The temporal route requires that order explicitly.

10. **How does change detection work here?**
    SatQuery sends the declared PRE and POST images to Chg2Cap, which returns a natural-language change description.

11. **Does the temporal route produce a change map?**
    No. It produces text only and does not claim a mask, bounding box, or area estimate.

12. **What is BigEarthNet used for?**
    Approved non-test BigEarthNet development inputs support the single-image and optical-SAR demonstrations.

13. **What is LEVIR-CC used for?**
    The admitted LEVIR-CC validation pair supports the PRE/POST change-description demonstration.

14. **Why is grounding blocked?**
    This release has no validated grounding specialist or trustworthy pixel-coordinate mapping, so it fails closed.

15. **How is confidence handled?**
    Where calibration is unavailable, SatQuery returns `NOT_AVAILABLE` instead of inventing a percentage.

16. **How does SatQuery reduce hallucination risk?**
    It validates input roles, exposes specialist provenance and warnings, and blocks unsupported routes instead of silently substituting another route.

17. **What happens with unsupported inputs?**
    The route-specific validator returns a structured error or blocked result before specialist execution.

18. **What happens with Cartosat-2S or RISAT?**
    They remain unvalidated. The system must not claim support until sensor-specific adapters and evidence are available.

19. **How is test leakage prevented?**
    The release uses approved development inputs only and maintains zero-access counters for test content, labels, inference, and metrics.

20. **Why did you avoid test data?**
    Development and demonstration must not contaminate held-out evaluation evidence.

21. **What are the single-image VQA results?**
    Internal panels recorded 17/30 binary and 8/30 MCQ. They are not public benchmark results.

22. **What is the optical-SAR result?**
    The joint route works technically, but the controlled comparison did not show improvement over the optical-only baseline.

23. **What is the temporal result?**
    The approved smoke panel generated nonempty captions for 5/5 development pairs. It is a functional check, not a benchmark metric.

24. **What is novel about the project?**
    Its contribution is the explicit specialist-routing and evidence contract across single-image, optical-SAR, temporal, and scene-description workflows, with safe blocking for unsupported grounding.

25. **How does the agent select a specialist?**
    It combines declared roles and modalities with query intent, then checks the selected capability’s input contract.

26. **What is provenance?**
    Provenance identifies the selected route, input summary, specialist/model details, and route-specific source information.

27. **What is the execution trace?**
    It is a public operational record of input analysis, route selection, validation, specialist execution, and result integration. It is not private chain-of-thought.

28. **Can the system work offline?**
    Yes, the verified release is a loopback local application when its approved local models and data are present.

29. **What GPU was used for final verification?**
    The final rehearsal used an NVIDIA GeForce RTX 5060 Laptop GPU with 8123 MiB reported VRAM under the validated CUDA runtime.

30. **What happens when CUDA is unavailable?**
    Model routes may be unavailable or slower. A CPU fallback must not be presented as the verified GPU demo without a separate validation.

31. **What is the project’s main limitation?**
    Evidence differs by route: VQA is internal-only, temporal output is text-only, scene description lacks a benchmark receipt, and grounding is unavailable.

32. **Why is this suitable for SIH?**
    It provides a concrete, transparent remote-sensing interaction workflow with multimodal and temporal routes, explicit evidence, and safe failure behavior.

33. **What would you improve next?**
    Authorized benchmark evaluations, validated grounding, spatial change masks, calibrated uncertainty, and validated Cartosat/RISAT adapters.
