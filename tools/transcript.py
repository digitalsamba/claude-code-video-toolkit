"""Normalize toolkit captions and Whisper/WhisperX segments to timed words."""
from __future__ import annotations

from media_ops import number


def segments(value):
    """Return validated utterances. Whisper timestamps are seconds, Caption[] is ms."""
    if isinstance(value, list):
        if not value:
            raise ValueError('Transcript cannot be empty')
        captions = []
        for word in value:
            if not isinstance(word, dict):
                raise ValueError('Each caption must be an object')
            captions.append({'start': number(word.get('startMs'), 'caption startMs', 0) / 1000,
                             'end': number(word.get('endMs'), 'caption endMs', 0) / 1000,
                             'text': word.get('text'), 'speaker': word.get('speaker', 'SPEAKER_00')})
        value = {'segments': captions}
    if not isinstance(value, dict) or not isinstance(value.get('segments'), list) or not value['segments']:
        raise ValueError('Use Caption[] or a Whisper/WhisperX object with segments')
    result = []
    previous = -1
    for index, item in enumerate(value['segments']):
        if not isinstance(item, dict) or not isinstance(item.get('text'), str) or not item['text'].strip():
            raise ValueError(f'Transcript segment {index} needs text')
        start = number(item.get('start'), 'segment start', 0)
        end = number(item.get('end'), 'segment end', start + .000001)
        if start < previous:
            raise ValueError('Transcript segments must be ordered by start')
        previous = start
        speaker = item.get('speaker', 'SPEAKER_00')
        if not isinstance(speaker, str) or not speaker.strip():
            raise ValueError('Speaker ID must be a nonempty string')
        words = []
        word_previous = start
        if not isinstance(item.get('words', []), list):
            raise ValueError('Transcript words must be an array')
        for word in item.get('words', []):
            if not isinstance(word, dict):
                raise ValueError('Transcript words must be objects')
            # WhisperX can emit unaligned words without timestamps. Do not invent them.
            if 'start' not in word or 'end' not in word:
                words = []
                break
            text = word.get('word', word.get('text'))
            if not isinstance(text, str) or not text.strip():
                raise ValueError('Transcript words need text')
            left = number(word['start'], 'word start', start, end)
            right = number(word['end'], 'word end', left + .000001, end + .001)
            if left < word_previous - .001:
                raise ValueError('Transcript word times must not overlap')
            word_previous = right
            words.append({'text': text.strip(), 'start': left, 'end': right})
        result.append({'start': start, 'end': end, 'text': item['text'].strip(),
                       'speaker': speaker, 'words': words})
    return result


def transcribe(path, model='base'):
    try:
        import whisper
    except ImportError as error:
        raise ValueError('Local transcription requires uv sync --extra whisper, or supply --transcript') from error
    engine = whisper.load_model(model)
    return engine.transcribe(str(path), word_timestamps=True, verbose=False, fp16=False)
