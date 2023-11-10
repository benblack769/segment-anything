import onnxruntime
import onnx
from PIL import Image
import numpy as np

urine_image = Image.open("dense.jpg")
urine_data = np.array(urine_image)

crop = urine_data[:1024,:1024].astype(np.float32)
means = np.array([123.675, 116.28, 103.53], dtype=np.float32)
stddevs = np.array([58.395, 57.12, 57.375], dtype=np.float32)
crop = (crop - means) / stddevs
crop = crop.transpose((2,0,1))
crop = np.expand_dims(crop, 0)

sess = onnxruntime.InferenceSession("vit_b_lm_encoder.onnx.bf16.onnx")

embeddings, = sess.run(None, {"input_image": crop})
print(embeddings.shape)
np.save("embeddings.npy",embeddings)
embeddings = np.load("embeddings.npy")
# exit(0)

sess = onnxruntime.InferenceSession("vit_b_lm_decoder.onnx")

embed_dim = embeddings.shape[1] # dimensions of the vit_b model
embed_size = embeddings.shape[2:]
np.random.seed(42)

mask_input_size = [4 * x for x in embed_size]
point_coords = np.random.randint(low=0, high=1024, size=(1, 5, 2)).astype( dtype=np.float32)
point_labels = np.random.randint(low=0, high=4, size=(1, 5)).astype( dtype=np.float32)
input_data = {
    "image_embeddings": embeddings,
    "point_coords": point_coords,
    "point_labels": point_labels,
    "mask_input": np.ones((1, 1, *mask_input_size), dtype=np.float32),
    "has_mask_input": np.array([1], dtype=np.float32),
    "orig_im_size": np.array([1024, 1024], dtype=np.float32),
}
# input_data = {key: value.numpy() for key, value in input_data.items()}
outs = sess.run(None, input_data)
print(len(outs))
print([out.shape for out in outs])
print(outs[1])