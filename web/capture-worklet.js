/* Turns the browser's 128-sample blocks into the 20ms frames the server wants.
 *
 * Runs on the audio thread, so it must not allocate per block or the capture
 * glitches and the transcript degrades in a way that looks like a bad model.
 *
 * Two jobs, and both are the reason this is a worklet rather than a
 * ScriptProcessor: downsample 48kHz to 16kHz, and buffer to a fixed frame size.
 * The browser delivers 128 samples at a time, which does not divide into a
 * 320-sample frame, so without buffering here every frame the server sees is
 * ragged and the audio drifts out of alignment.
 */

const TARGET_RATE = 16000;
const FRAME_SAMPLES = 320; // 20ms at 16kHz

class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / TARGET_RATE;
    this.frame = new Int16Array(FRAME_SAMPLES);
    this.filled = 0;
    // Fractional read position into the incoming block. Carried across blocks,
    // because dropping the remainder each time is how a resampler slowly gains
    // a semitone.
    this.position = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const channel = input[0];

    while (this.position < channel.length) {
      // Nearest neighbour. Good enough for speech at this ratio, and a proper
      // polyphase filter costs CPU on the audio thread for quality no
      // transcription model will notice.
      const sample = channel[Math.floor(this.position)];
      const clamped = Math.max(-1, Math.min(1, sample));
      this.frame[this.filled++] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
      this.position += this.ratio;

      if (this.filled === FRAME_SAMPLES) {
        // Copied, because the port transfers and this buffer keeps being written.
        this.port.postMessage(this.frame.slice());
        this.filled = 0;
      }
    }

    this.position -= channel.length;
    return true;
  }
}

registerProcessor("capture", CaptureProcessor);
