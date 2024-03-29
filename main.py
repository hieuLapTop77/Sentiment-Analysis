import numpy as np
import pandas as pd
import pickle
import streamlit as st
import regex
from underthesea import word_tokenize, pos_tag, sent_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from streamlit_folium import folium_static
import folium
import time
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud
########################################################################

positive_words = [
    "thích", "tốt", "xuất sắc", "tuyệt vời", "tuyệt hảo", "đẹp", "ổn", "ngon",
    "hài lòng", "ưng ý", "hoàn hảo", "chất lượng", "thú vị", "nhanh", "đúng"
    "tiện lợi", "dễ sử dụng", "hiệu quả", "ấn tượng",
    "nổi bật", "tận hưởng", "tốn ít thời gian", "thân thiện", "hấp dẫn",
    "gợi cảm", "tươi mới", "lạ mắt", "cao cấp", "độc đáo",
    "hợp khẩu vị", "rất tốt", "rất thích", "tận tâm", "đáng tin cậy", "đẳng cấp",
    "hấp dẫn", "an tâm", "không thể cưỡng lại", "thỏa mãn", "thúc đẩy",
    "cảm động", "phục vụ tốt", "làm hài lòng", "gây ấn tượng", "nổi trội",
    "sáng tạo", "quý báu", "phù hợp", "tận tâm",
    "hiếm có", "cải thiện", "hoà nhã", "chăm chỉ", "cẩn thận",
    "vui vẻ", "sáng sủa", "hào hứng", "đam mê", "vừa vặn", "đáng tiền", "xứng đáng", "đầy đủ", "đủ",'ổn', 'kỹ', 'gần', 'vui vẻ', 'hài lòng', 'yên tâm', 'rẻ', 'rõ ràng', 'cẩn thận', 'dc', 'thơm',
    'xuất sắc', 'thoải mái', 'nóng hổi', 'đặc biệt', 'đẹp', 'rộng rãi', 'sạch sẽ', 'hiện đại', 'mạnh', 'tốt', 'chắc chắn', 'đều đặn', 'nhanh', 'tiện lợi', 'ổn định', 'rộng lớn', 'mát', 'ok', 'nhiệt tình', 'lẹ',
    'niềm nở', 'xinh xắn', 'tự nhiên', 'chuyên nghiệp', 'lịch sự', 'đàng hoàng', 'thuận lợi', 'ấn tượng', 'tiện', 'đa dạng', 'thoáng mát', 'đẹp mắt', 'hợp lý', 'dễ thương', 'tuyệt vời', 'ấm cúng',
    'dịu', 'nhẹ', 'xinh gái', 'tử tế', 'nhẹ nhàng', 'nhanh nhẹn', 'nhanh chóng', 'gọn', 'đậm đà', 'mị thít', 'thơm ngậy', 'đặc sắc', 'ưng', 'nổi bật', 'xịn', 'thoáng', 'ăn ok', 'sướng', 'khỏi phải nói', 'hoành tráng',
    'ấm', 'no á', 'tận tình', 'nhah', 'lạ', 'đông đúc', 'đắc địa', 'bình dân', 'chuyện nghiệp', 'bắt vị', 'tấp nập', 'mau', 'đậm vị', 'giòn nha', 'dễ chịu', 'ko bở', 'bự', 'dễ tính', 'đầy đặn', 'hấp dẫn', 'ngon miệng', 'mát mẻ',
    'chiên giòn', 'tuyệt', 'sạch', 'miệt mài', 'thích vị', 'giòn rụm', 'cảm kích', 'ngọt nha', 'sáng sủa', 'tươi ngon',
    'lễ phép', 'gọn gàng', 'đầy ắp', 'ổn áp', 'chỉn chu', 'siêu to', 'khổng lồ', 'ngon lành', 'kha khá', 'thỏa mãn', 'phi thơm', 'chu đáo', 'an toàn', 'kỹ lưỡng', 'nhộn nhịp', 'vừa phải', 'bún ngon', 'bún đẹp', 'quen thuộc',
    'ngon mắt', 'ưng quán', 'lạ miệng', 'an tâm', 'chất lượng', 'gần gũi', 'tuyệt', 'tận tay', 'bbq thơm', 'mềm thơm', 'xốp giòn', 'phải chăng', 'chuẩn', 'salad tươi', 'hợp', 'tuyệt hương vị',
    'no no', 'hoàn chỉnh', 'đượm vị', 'thượng hạng', 'ăn hợp', 'tốt nha', 'độc đáo', 'xinh', 'bắt mắt', 'ổn hen', 'rộp rộp', 'ưu điểm', 'giải nhiệt',
    'ok nha', 'chính xác', 'hợp miệng', 'ngon mềm', 'hút hồn', 'hòa quyện', 'hoàn hảo', 'khá', 'ngon nhaaa', 'hợp vị', 'mát nè', 'lãng mạn', 'thông thoáng', 'chỉnh chu', 'vừa vị', 'mới lạ',
    'trọn vẹn', 'xịn sò', 'dễ ăn', 'sang trọng', 'ân cần', 'ngon nè', 'giòn tan', 'yên tĩnh', 'thanh bình', 'thích hợp', 'oke', 'ngon nha', 'nice', 'mềm mại', 'á ngon', 'siêu ngon', 'khoái',
    'ưu tiên', 'thông minh', 'ăn thơm', 'nườm nượp', 'nổi danh', 'vừa ý', 'gà giòn', 'cute', 'hào hứng', 'sành điệu', 'đỉnh', 'cuteee', 'ghiền', 'đậm nét', 'lan tỏa', 'khéo léo', 'cảm tình',
    'ưng ý', 'tươm tất', 'nhiệt huyết', 'thuận tiện', 'bát mắt', 'điểm cộng', 'good', 'xinh xinh', 'tuyệt luôn', 'giỏi', 'thơm lừng', 'tươi xanh', 'ngon cực', 'ngon đặc sắc', 'niềm nỡ',
    'best', 'ăn bá chấy', 'ăn ngon', 'thơm thơm', 'thân thương', 'miễn chê', 'thích thú', 'ngon lâu', 'náo nhiệt', 'đẳng cấp', 'toẹt vời', 'thỏa sức', 'cuống', 'dễ thương nha', 'bao phê', 'mát lạnh',
    'vượt trội', 'xuất xắc', 'ngon tẹo', 'mát rượi', 'bổ dưỡng', 'vui nhộn', 'ăn mê', 'ô kê', 'phái', 'ngập mặt', 'kĩ nhé', 'sinh động', 'cưnng xĩu', 'siêu hợp', 'tiện nghi', 'bán chạy',
    'tuyêt vời', 'ngon bổ', 'trái cây ăn tươi', 'duoc', 'tinh khiết', 'siêu xịn', 'đồ sộ'
]

negative_words = [
    "kém", "tệ", "đau", "xấu", "dở", "ức", "nhỏ"
    "buồn", "rối", "thô", "lâu", "chán"
    "tối", "chán", "ít", "mờ", "mỏng",
    "lỏng lẻo", "khó", "cùi", "yếu",
    "kém chất lượng", "không thích", "không thú vị", "không ổn",
    "không hợp", "không đáng tin cậy", "không chuyên nghiệp",
    "không phản hồi", "không an toàn", "không phù hợp", "không thân thiện", "không linh hoạt", "không đáng giá",
    "không ấn tượng", "không tốt", "chậm", "khó khăn", "phức tạp",
    "khó hiểu", "khó chịu", "gây khó dễ", "rườm rà", "khó truy cập",
    "thất bại", "tồi tệ", "khó xử", "không thể chấp nhận", "tồi tệ","không rõ ràng",
    "không chắc chắn", "rối rắm", "không tiện lợi", "không đáng tiền", "chưa đẹp", "không đẹp", "tanh", "lâu",'dở', 'khó chịu', 'ngang', 'tệ', 'mau ngán', 'thiếu', 'ít', 'bở', 'cẩu thả', 'bình thường', 'ngập ngụa', 'nghèo',
    'tởm lợm', 'nhỏ xíu', 'hằn học', 'lười', 'kém', 'cợt nhã', 'mặn', 'vắng', 'nguội', 'ồn ào', 'ồn í', 'thú vị', 'bất tiện', 'ghê', 'đắt', 'khó', 'già', 'nhạt', 'lạnh', 'nhạt nhẽo', 'hỏng', 'xấc xược', 'bực mình',
    'lơ ngơ', 'nồng', 'mù', 'nhanh ngấy', 'dơ', 'hôi', 'ngán', 'sốc', 'ồn', 'khó tính', 'ức chế', 'quá kì', 'chật', 'hư', 'bất xúc', 'ngộp', 'nghi ngút', 'lâu lâu', 'chứ tệ', 'thúi', 'buồn', 'kinh', 'dai', 'tệ hại',
    'chật chội', 'lòng vòng', 'bố đời', 'ghê gớm', 'đắt đỏ', 'tụt hứng', 'giả', 'kì kì', 'ko hợp', 'sần', 'hắc ám', 'mề', 'ko thơm', 'xui', 'ngậy ngậy', 'bất thường', 'review tệ', 'lỏng lẻo', 'rách nát',
    'hụt hẫng', 'chậm', 'nhão', 'nhỏ hẹp', 'hẹp', 'mệt', 'bức xúc', 'ko hảo', 'nát', 'nhược điểm','vội', 'dầu mỡ', 'lơ là', 'gớm', 'mặn chát', 'mất hứng', 'lóng ngóng', 'mỡ', 'chật hẹp', 'khinh thường', 'lộn xộn',
    'mòn mỏi', 'rách', 'ngột ngạt','cộc lốc', 'cua bể', 'vô lễ', 'cực tệ', 'uể oải', 'bất lịch sự', 'dai', 'nhanh ngán', 'hấp dẩn', 'vô duyên', 'ngon ạ', 'gò bó', 'thối', 'khô sáp', 'đắng', 'gà hôi', 'bất hợp lý', 'xấu', 'gấp gáp',
    'hăng', 'tan nát', 'sợ hãi', 'lúng túng', 'hề tanh', 'cũ kỹ', 'ồn ao', 'fail', 'nhức nhối', 'chảnh', 'lồi lõm', 'mất dạy', 'thất vọng xíu', 'gian dối', 'khủng khiếp', 'ngán nhân viên', 'vụng về', 'choáng',
    'tồi tệ', 'giễu cợt', 'vô trách nhiệm', 'trầy trật', 'thiếu thiện cảm', 'chảnh chọe', 'kinh dị', 'xéo sắc', 'dơ dơ', 'sượng', 'bẩn bẩn', 'nguệch ngoạc', 'ngáo ngơ'
]

## english-vnmese
file = open('files/english-vnmese.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
english_vnmese = {}
for line in teen_lst:
    key, value = line.split('\t')
    english_vnmese[key] = str(value)
file.close()

#LOAD TEENCODE
file = open('files/teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
file.close()

### vietnamese-stopwords
file = open('files/vietnamese-stopwords.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()

##LOAD EMOJICON
file = open('files/emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()

#Load models 
model_pkl_file = 'model_rdf.pkl'


# List functions for handling data
def process_text(text: str, emoji_dict, teen_dict, english_vnmese):
  document = text.lower()
  document = document.replace("’",'')
  document = regex.sub(r'\.+', ".", document)
  new_sentence =''
  for sentence in sent_tokenize(document):
    sentence = regex.sub(r'(?<=[^\W\d_])\b', ' ', sentence)
    sentence = ''.join(emoji_dict[word]+' ' if word in emoji_dict else word for word in list(sentence))
    sentence = ' '.join(english_vnmese[word] if word in english_vnmese else word for word in sentence.split())
    sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())
    ###### DEL Punctuation & Numbers
    pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
    sentence = ' '.join(regex.findall(pattern,sentence))
    new_sentence = new_sentence+ sentence + '. '
  document = new_sentence
  document = regex.sub(r'\s+', ' ', document).strip()
  return document

def loaddicchar():
    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic

def covert_unicode(txt):
    dicchar = loaddicchar()
    return re.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)

def process_special_word(text):
  new_text = ''
  text_lst = text.split()
  i= 0
  if 'không' in text_lst:
    while i <= len(text_lst) - 1:
      word = text_lst[i]
      if word == 'không':
        next_idx = i+1
        if next_idx <= len(text_lst) -1:
          word = word +'_'+ text_lst[next_idx]
        i= next_idx + 1
      else:
        i = i+1
      new_text = new_text + word + ' '
  else:
    new_text = text
  return new_text.strip()

def normalize_repeated_characters(text):
  return re.sub(r'(.)\1+', r'\1', text)

def process_postag_thesea(text):
  new_document = ''
  for sentence in sent_tokenize(text):
    sentence = sentence.replace('.','')
    lst_word_type = ['N','Np','A','AB','V','VB','VY','R']
    sentence = ' '.join( word[0] if word[1].upper() in lst_word_type else '' for word in pos_tag(process_special_word(word_tokenize(sentence, format="text"))))
    new_document = new_document + sentence + ' '
  new_document = regex.sub(r'\s+', ' ', new_document).strip()
  return new_document

def remove_stopword(text, stopwords):
  document = ' '.join('' if word in stopwords else word for word in text.split())
  document = regex.sub(r'\s+', ' ', document).strip()
  return document

def find_words(document, list_of_words):
  document_lower = document.lower()
  word_count = 0
  word_list = []
  for word in list_of_words:
    if word in document_lower:
      # print(word)
      word_count += document_lower.count(word)
      word_list.append(word)
  return word_count



######################################
def handle_comment(cmt):
    document = process_text(cmt, emoji_dict, teen_dict, english_vnmese)
    document = covert_unicode(document)
    document = process_special_word(document)
    document = normalize_repeated_characters(document)
    document = process_postag_thesea(document)
    document = remove_stopword(document, stopwords_lst)
    pos = find_words(document, positive_words)
    neg = find_words(document, negative_words)
    vectorizer = TfidfVectorizer(max_features=1000)
    comment = vectorizer.fit_transform([document])
    tfidf_array_full = np.zeros((comment.shape[0], 1000))
    tfidf_array_full[:, :comment.shape[1]] = comment.toarray()
    df_comment = pd.DataFrame(tfidf_array_full)
    df_comment['positive_words'] = pos
    df_comment['negative_words'] = neg
    df_comment.columns = df_comment.columns.astype(str)
    return df_comment

def predict(model_pkl_file, id = None, comment = None):
    df_new = pd.read_csv('files/matrix_comment.csv.gz', compression='gzip')
    try:
        df_new = df_new.drop(['Unnamed: 0'], axis=1)
    except KeyError:
       pass
    cmt = None
    id_res = None
    with open(model_pkl_file, 'rb') as file:
        loaded_model = pickle.load(file)
    if id is not None and comment is None:
        if id in df_new['ID'].unique().tolist():
            id_res = df_new.loc[df_new['ID'] == id].drop(columns=['ID'])
        else:
            print('Not found restaurant')
        prediction = loaded_model.predict(id_res)
    else:
        cmt = handle_comment(comment)
        prediction = loaded_model.predict(cmt)
    return prediction

def get_pos(lat,lng):
    return lat,lng

def get_id_input():
    id = None
    while id is None:
        id = st.number_input("ID Restaurant from 1 to 1621", min_value=1, max_value=1621)
    return id


def main():
    # 1. Read data
    st.title("Data Science Project")
    st.write("## Sentiment Analysis")
  
    menu = ["Thuật toán", "Thông tin về thuật toán", "About"]
    choice = st.sidebar.selectbox('Menu', menu)
    if choice == 'Thuật toán':    
        df = pd.read_csv('files/Restaurants.csv')
        df_comment = pd.read_csv('files/2_Reviews.csv')
        st.subheader("Rating classification")
        on = st.toggle('ID Restaurant')
        if on:
            map_vietnam = folium.Map(location=[10.809929141198806, 106.64572837501036], zoom_start=10)
            map_vietnam.add_child(folium.LatLngPopup())
            choose = st.radio('Chọn phương thức', ['Nhập ID nhà hàng', 'Chọn ID nhà hàng'])
            id = None
            if choose == 'Nhập ID nhà hàng':
                id = get_id_input()
            else:
                id = st.selectbox('Vui lòng chọn ID nhà hàng', tuple(df['ID']))
            

            result = {
               "ID" : id,
               "Name": df.loc[df['ID']==id]['Restaurant'],
               "Price": df.loc[df['ID']==id]['Price'],
               "Rating": str(df_comment[df_comment['IDRestaurant'] == id]['Rating'].mean()) + '/10'
            }
            data = pd.DataFrame(result)
            st.dataframe(data)
            st.write('Một vài bình luận')
            comment_pos = []
            pos = []
            idx_pos = []
            comment_neg = []
            neg = []
            idx_neg = []
            try:
                # df_ = df_comment[df_comment['IDRestaurant'] == id]['Comment'][:3].to_frame()
                list_index = df_comment[df_comment['IDRestaurant'] == id]['Comment'].index
                output = predict(model_pkl_file, id = id)
                for i in range(len(output)):
                    if output[i] == 0 or output[i] == '0':
                        if len(df_comment['Comment'][list_index[i]]) > 0:
                            comment_neg.append(df_comment['Comment'][list_index[i]])
                        else:
                           comment_neg.append("")
                        neg.append("Tiêu cực")
                        idx_neg.append(list_index[i])
                    else:
                        if len(df_comment['Comment'][list_index[i]]) > 0:
                            comment_pos.append(df_comment['Comment'][list_index[i]])
                        else:
                           comment_pos.append("")
                        pos.append("Tích cực")
                        idx_pos.append(list_index[i])

                result_pos = {
                   "ID": idx_pos,
                   "Comment": comment_pos,
                   "Rating": pos
                }

                df_pos = pd.DataFrame(result_pos)
                st.dataframe(df_pos[:11])
                result_neg = {
                   "ID": idx_neg,
                   "Comment": comment_neg,
                   "Rating": neg
                }
                
                df_neg = pd.DataFrame(result_neg)
                st.dataframe(df_neg[:11])
            except Exception:
               pass

            folium.Marker([df.loc[id]['latitude'], df.loc[id]['longitude']]).add_to(map_vietnam)
            try:
            
                text = ' '.join(df_comment[df_comment['IDRestaurant'] == id]['Comment'])
                wordcloud = WordCloud(width=800, height=400, random_state=42, background_color='white', max_words=40).generate(text)

                # Trực quan hóa WordCloud
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('Word Cloud - Most Frequent Words in Comments')
                plt.show()
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()
            except Exception:
               pass
            folium_static(map_vietnam)

        else:
            
            map_vietnam = folium.Map(location=[10.809929141198806, 106.64572837501036], zoom_start=10)
            map_vietnam.add_child(folium.LatLngPopup())
            for index, row in df.iterrows():
                folium.Marker([row['latitude'], row['longitude']]).add_to(map_vietnam)

            # # Hiển thị bản đồ trong Streamlit
            st.subheader('Các nhà hàng tại Thành phố Hồ Chí Minh')
            folium_static(map_vietnam)
            title = st.text_input("Comment")
            if len(title) > 0:
                output = predict(model_pkl_file, comment = title)
                if output[0] == 0 or output[0] == '0':
                    st.write(f'Bình luận {title} này là tiêu cực')
                else:
                    st.write(f'Bình luận {title} này là tích cực')


    elif choice == 'Thông tin về thuật toán':
        st.image('3.PNG', caption='Mức độ phân bố của các nhà hàng')
        st.image('2.PNG', caption='Mức độ phân bố rating của nhà hàng')
        st.write("Dựa vào biểu đồ trên chúng ta có thể thấy rằng tập dữ liệu này có một mức độ mất cân bằng đáng kể giữa các nhãn rating.\
                 Với khoảng rating 6-10 chiếm 2/3 tập dữ liệu và chỉ 1/3 mẫu trong khoảng rating 0-6, chúng ta thấy rằng có một sự chênh lệch lớn giữa hai nhóm này. \
                 Điều này có thể gây ra các vấn đề khi chúng ta huấn luyện mô hình machine learning.")
        st.image('1.PNG', caption='So sánh thời gian và độ chính xác của các thuật toán')
        st.subheader("Classification Report")
        json_data = """
        {
            "0": {
                "precision": 0.87,
                "recall": 0.89,
                "f1-score": 0.88,
                "support": 3944
            },
            "1": {
                "precision": 0.89,
                "recall": 0.87,
                "f1-score": 0.88,
                "support": 3927
            },
            "accuracy": {
                "precision": "",
                "recall": "",
                "f1-score": 0.88,
                "support": 7871
            },
            "macro avg": {
                "precision": 0.88,
                "recall": 0.88,
                "f1-score": 0.88,
                "support": 7871
            },
            "weighted avg": {
                "precision": 0.88,
                "recall": 0.88,
                "f1-score": 0.88,
                "support": 7871
            }
        }
        """

        # Load JSON data
        report_data = json.loads(json_data)
        data = {}
        for key, value in report_data.items():
            if isinstance(value, dict):
                data[key] = [value["precision"], value["recall"], value["f1-score"], value["support"]]
            else:
                data[key] = [value]  # Đảm bảo giá trị là một danh sách

        # Chuyển đổi dữ liệu thành DataFrame
        df = pd.DataFrame.from_dict(data, orient='index', columns=["precision", "recall", "f1-score", "support"])

        # Hiển thị DataFrame trên Streamlit
        st.dataframe(df)

    elif choice == 'About':
        st.subheader("Personal Information")
        st.write("Designed by Nguyen Ba Dinh")
        st.write("Phone: 0983879645")
        st.write("Nhậu tốt, gáy to")
        

if __name__ == "__main__":
   main()

