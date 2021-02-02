import chardet
import re


def lang_is_en(x):
    x = x.encode('utf-8')
    if len(x) <= 3:
        return False
    return chardet.detect(x)['language'] == ''


def clean_df(df):
    df = df.fillna(0)
    df = df[['id', 'title', 'post', 'sub_score', 'score_esh', 'score_yta', 'score_nta']]
    df = df[df['post'].str.strip() != '']
    df.columns = [i.replace('score_', '') for i in df.columns]
    df['condensed_class'] = df[['yta', 'nta']].idxmax(axis=1)
    df = df[df['post'].str.strip() != '[deleted]']
    df = df[df['post'].fillna('abcdefg').astype(str).str[:80].apply(lang_is_en)]
    df['encoded_class'] = df['condensed_class'].map({'nta': 0, 'yta': 1})
    return df


mypattern = re.compile(r'[^a-zA-z\s]')
mypattern2 = re.compile(r'\s+')

def clean_text(x):
    x = x.lower()
    text = mypattern.sub('', x)
    text = mypattern2.sub(' ', text)
    return text


def clean_numbers(x):
    # if bool(re.search(r'\d', x)):
    # x = re.sub('[0-9]{5,}', '#', x)
    # x = re.sub('[0-9]{4}', '#', x)
    # x = re.sub('[0-9]{3}', '#', x)
    # x = re.sub('[0-9]{2}', '#', x)
    x = re.sub('[0-9]+', '#', x)
    return x


def large_clean(x):
   x = clean_text(x)
   x = clean_numbers(x)
   return x