num_feats = {
    'ml-1m': 15226,
    'book-crossing': 110000,
    'ali': 232578,
    'ad': 3029332,
    'eleme': 16516884,
    'book': 6213153

}

padding_idxs = {
    'ad':3029332,
    'eleme':0,
    'book':6213153
}

num_fields = {
    'ml-1m': 14,
    'book-crossing': 39,
    'ali': 14,
    'ad': 12,
    'eleme': 17,
    'book': 15
}

item_fields = {
    'ad': 4,
    'eleme': 9,
    'book': 12
}

hist_fields = {
    'ad': 2,
    'eleme': 8,
    'book': 12
}

patterns = {
    'ad': ['user_cate_pid.pkl', 'user.pkl', 'gender_age.pkl', 'user_attrs.pkl']
}

pattern_keys = {
    'ad': [[0, 8, 11], [0], [3, 4], [1, 3, 4, 5, 6]]
}

names = {
    'ad': ['user_cate_pid', 'user', 'gender_age', 'user_attrs']
}
