# Demo Script

## Short intro
“This is a proof-of-concept for AI-assisted signature verification. The goal is not to make a production fraud decision, but to show how a system can compare a reference signature against a questioned signature, generate a similarity score, and route the result as match, review, or mismatch.”

## What and why
“The idea is simple: preprocess the signatures, convert them into embeddings with a Siamese neural network, and compare their distance. Lower distance means the signatures are more similar. In a real engagement, this would support operations review, not replace governance or validation.”

## Demo workflow
“Start with the first example in the Examples section to show a likely match.”
“Click Compare and point out the processed signature crops, the similarity score, and the green verdict.”
“Then load the mismatch example to show how a different writer produces a lower similarity score and a different verdict.”
“If needed, mention that borderline cases would be routed to review rather than forced into a binary decision.”
## Short close
“What this demonstrates is the workflow: image in, preprocessing, model comparison, score, and verdict. What it does not demonstrate is production calibration, enterprise error rates, or bank-grade deployment. Those would come in a pilot with real client data.”

