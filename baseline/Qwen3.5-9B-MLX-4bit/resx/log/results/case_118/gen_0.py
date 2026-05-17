
def get_closest_vowel(word):
    """You are given a word. Your task is to find the closest vowel that stands between 
    two consonants from the right side of the word (case sensitive).
    
    Vowels in the beginning and ending doesn't count. Return empty string if you didn't
    find any vowel met the above condition. 

    You may assume that the given string contains English letter only.

    Example:
    get_closest_vowel("yogurt") ==> "u"
    get_closest_vowel("FULL") ==> "U"
    get_closest_vowel("quick") ==> ""
    get_closest_vowel("ab") ==> ""
    """
    vowels = set("aeiouAEIOU")
    consonants = set("bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ")
    
    # Iterate through the word from right to left
    for i in range(len(word) - 1, 0, -1):
        if word[i] in vowels:
            # Check if the character before and after are consonants
            if word[i-1] in consonants and word[i+1] in consonants:
                return word[i]
    
    return ""

def get_closest_vowels_concat(words):
    """You are given a list of words. Your task is to find the closest vowel for each word that stands between two consonants from the right side of the word (case sensitive). Then, concatenate all the vowels found in the order of the words and return the

assert get_closest_vowels_concat(['yogurt', 'FULL', 'quick', 'ab']) == 'uu__'
assert get_closest_vowels_concat(['strength', 'apple', 'orange']) == 'e_a'
assert get_closest_vowels_concat(['sky', 'tree', 'book']) == '___'
assert get_closest_vowels_concat(['python', 'java', 'ruby']) == 'oau'