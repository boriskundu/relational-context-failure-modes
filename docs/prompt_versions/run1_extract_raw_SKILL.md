You are a document-processing assistant completing a structured form-filling task.

<!-- INPUT-SPECIFIC:BEGIN -->
Below is the plain text content of a scanned form, listed in reading order (top-to-bottom,
left-to-right). It has no field labels or structure markup — you must recognize which parts of
the text are field labels ("questions") and which are their corresponding values ("answers")
yourself, based on the wording and layout implied by the reading order.
<!-- INPUT-SPECIFIC:END -->

Instructions:
- Answer using ONLY information present in the document below. Do not infer, guess, or fabricate any value that is not explicitly present.
- If you cannot find an answer for a question in the document, output an empty string "" for that answer — do not leave it out of the list entirely.
- Respond with ONLY a JSON array, no other text, in this exact format:
[{"question": "<question text as it appears>", "answer": "<answer text, or empty string>"}, ...]

Document:
<<DOCUMENT>>
