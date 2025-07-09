A flask server app that receives a pdf file and a signature image in json.
Retrun if any signatures found above similarity score threshold.

- Convert pdf to image
- Detect signatures with pretrained yolo model
- Extract signatures and preprocess (resize, remove stamp, clean noise)
- Create embeddings and calculate similarities
- Return the result in json

Signature detection: YOLO
Embeddings: ResNet
Connected component analysis: OpenCV
