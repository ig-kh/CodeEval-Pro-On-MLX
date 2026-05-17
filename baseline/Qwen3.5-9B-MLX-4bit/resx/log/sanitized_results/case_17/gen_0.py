from typing import List


def parse_music(music_string: str) -> List[int]:
    """ Input to this function is a string representing musical notes in a special ASCII format.
    Your task is to parse this string and return list of integers corresponding to how many beats does each
    not last.

    Here is a legend:
    'o' - whole note, lasts four beats
    'o|' - half note, lasts two beats
    '.|' - quater note, lasts one beat

    >>> parse_music('o o| .| o| o| .| .| .| .| o o')
    [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]
    """
    from typing import List

    beats_map = {
        'o': 4,
        'o|': 2,
        '.|': 1
    }

    result = []
    current_note = ''

    for char in music_string:
        if char == ' ':
            result.append(beats_map.get(current_note, 0))
            current_note = ''
        else:
            current_note += char

    if current_note:
        result.append(beats_map.get(current_note, 0))

    return result


def total_beats(music_measures: List[str]) -> int:
    """ Given a list of musical note strings, each representing a measure in a piece of music, calculate the total number of beats for each measure and return the sum of all beats in the piece. Each measure is separated by a space and the notes within a measure are also separated by a space. Use the parse_music function to parse each measure and sum the beats.
    """
    total_beats = 0
    for measure in music_measures:
        beats = parse_music(measure)
        total_beats += sum(beats)

    return total_beats


assert total_beats(['o o| .|', 'o| o| .| .|', 'o o']) == 21
assert total_beats(['o| .| o| .|', 'o o o']) == 18
assert total_beats(['.| .| .| .|', 'o o| o| o']) == 16