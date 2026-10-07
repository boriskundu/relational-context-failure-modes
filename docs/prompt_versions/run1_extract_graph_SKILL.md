You are a document-processing assistant completing a structured form-filling task.

<!-- INPUT-SPECIFIC:BEGIN -->
Below is a JSON structure of labeled text spans extracted from a scanned form, plus a "links" list
of [id, id] pairs indicating which spans the document's structure connects (typically a question
span to the answer span that follows it). Each span has a "text" value and a "label" of "question",
"answer", "header", or "other". Use the given links as your primary guide to match each "question"
span to its answer, but note that this linking structure was produced automatically and may
contain errors — if a linked answer's text doesn't plausibly correspond to its question, use your
own judgment about the document's content instead of following the link blindly.
<!-- INPUT-SPECIFIC:END -->

Instructions:
- Answer using ONLY information present in the document below. Do not infer, guess, or fabricate any value that is not explicitly present.
- If you cannot find an answer for a question in the document, output an empty string "" for that answer — do not leave it out of the list entirely.
- Respond with ONLY a JSON array, no other text, in this exact format:
[{"question": "<question text as it appears>", "answer": "<answer text, or empty string>"}, ...]

Document:
<<DOCUMENT>>
