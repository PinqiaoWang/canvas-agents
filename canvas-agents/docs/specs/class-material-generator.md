# Class Material Generator — Feature Spec (MVP)

**Goal**: Generate a ready-to-teach packet grounded in course materials (PDF/slides/pages/modules).

## Inputs
- course_id
- topic (or module_title)
- target_duration_min (e.g., 50)
- audience_level (intro / intermediate / advanced)
- constraints (optional; e.g., "2 worked examples")

## Output
Return `TeachingPacket` JSON + rendered Markdown.

Required sections:
1) Learning objectives (Bloom)
2) Concept map / key terms
3) Time-boxed lecture outline
4) Worked examples (step-by-step)
5) In-class questions (easy→hard)
6) Practice problems + solutions
7) Common confusions + fixes
8) Sources/citations (file + page/slide + module context when available)

## LangGraph nodes (MVP)
1. Router (intent + constraints)
2. Retriever (Pinecone top-k, diversify across files)
3. Extractor (ConceptSet JSON)
4. Planner (Bloom + time boxes)
5. Composer + Verifier (TeachingPacket JSON + citation checks)

## Definition of Done
- Works for 2 real topics in a sandbox course
- Every course-derived claim has ≥1 citation
- Output schema validates
