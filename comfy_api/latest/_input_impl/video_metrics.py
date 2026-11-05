"""Bounded scalar work observations for the native file decoder."""
from comfy import execution_metrics


class VideoDecodeMetrics:
    def __init__(self, full_decode, audio_stream):
        self.complete = True
        self.video_signature = self.audio_signature = None
        self.values = dict(fullDecode=full_decode, audioStream=audio_stream, videoFrames=0, videoPixels=0,
                           processedFrames=0, processedPixels=0, convertedPixels=0, alignmentPixels=0,
                           rotatedFrames=0, croppedFrames=0, resizedFrames=0, audioFrames=0, audioSamples=0,
                           resampledAudioFrames=0, resampledAudioSamples=0, decodeErrors=0)

    def streams(self, video, audio, version):
        try:
            self.values.update(videoCodec=video.codec.canonical_name, decoderVersion=version)
            if audio is not None:
                self.values['audioCodec'] = audio.codec.canonical_name
        except Exception:
            self.complete = False

    def video(self, frames):
        try:
            for frame in frames:
                signature = (frame.width, frame.height, frame.format.name, max(component.bits for component in frame.format.components))
                if self.video_signature is None:
                    self.video_signature = signature
                    self.values.update(videoWidth=signature[0], videoHeight=signature[1], videoFormat=signature[2], videoBitDepth=signature[3])
                elif signature != self.video_signature:
                    self.complete = False
                self.values['videoFrames'] += 1
                self.values['videoPixels'] += frame.width * frame.height
        except Exception:
            self.complete = False

    def processed(self, frame, shape, conversion, aligned, rotation, cropped, resized):
        try:
            previous = self.values.get('conversionFormat')
            if previous is not None and previous != conversion:
                self.complete = False
            self.values['conversionFormat'] = conversion
            self.values['processedFrames'] += 1
            self.values['processedPixels'] += shape[0] * shape[1]
            self.values['convertedPixels'] += frame.width * frame.height
            self.values['alignmentPixels'] += ((frame.width + 31) // 32) * 32 * ((frame.height + 31) // 32) * 32 if aligned else 0
            self.values['rotatedFrames'] += bool(rotation)
            self.values['croppedFrames'] += cropped
            self.values['resizedFrames'] += resized
        except Exception:
            self.complete = False

    def audio(self, frames):
        try:
            for frame in frames:
                signature = (frame.sample_rate, len(frame.layout.channels), frame.format.name)
                if self.audio_signature is None:
                    self.audio_signature = signature
                    self.values.update(audioRate=signature[0], audioChannels=signature[1], audioFormat=signature[2])
                elif signature != self.audio_signature:
                    self.complete = False
                self.values['audioFrames'] += 1
                self.values['audioSamples'] += frame.samples
        except Exception:
            self.complete = False

    def resampled(self, frame):
        try:
            self.values['resampledAudioFrames'] += 1
            self.values['resampledAudioSamples'] += frame.samples
        except Exception:
            self.complete = False

    def failed_packet(self):
        self.values['decodeErrors'] += 1
        self.complete = False

    def finish(self, images, alpha, audio):
        try:
            self.values['alphaFrames'] = alpha.shape[0] if alpha is not None else 0
            self.values['retainedAudioSamples'] = audio['waveform'].shape[-1] if audio is not None else 0
            self.complete = self.complete and self.values['processedFrames'] == images.shape[0]
            execution_metrics.emit('video_decode_work', version=1, complete=self.complete, **self.values)
        except Exception:
            execution_metrics.emit('video_decode_work', version=1, complete=False)
