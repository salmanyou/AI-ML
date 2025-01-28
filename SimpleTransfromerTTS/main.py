import IPython
from matplotlib import pyplot as plt
import torch

from model import TransformerTTS
from melspecs import inverse_mel_spec_to_wav
from text_to_seq import text_to_seq
from write_mp3 import write_mp3

# Define configuration parameters
hp = {
    'sr': 22050  # Sample rate, adjust as needed
}

train_saved_path = "train_SimpleTransfromerTTS.pt "
state = torch.load(train_saved_path)
model = TransformerTTS().cuda()
model.load_state_dict(state["model"])

text = "Hello, World."
name_file = "hello_world.mp3"

postnet_mel, gate = model.inference(
  text_to_seq(text).unsqueeze(0).cuda(),
  gate_threshold=1e-5,
  with_tqdm=False
)

audio = inverse_mel_spec_to_wav(postnet_mel.detach()[0].T)

plt.plot(
    torch.sigmoid(gate[0, :]).detach().cpu().numpy()
)

write_mp3(
    audio.detach().cpu().numpy(),
    name_file
)

IPython.display.Audio(
    audio.detach().cpu().numpy(),
    rate=hp['sr']  # Use sample rate from hp configuration
)
