import spacy
from spacy.training.example import Example
from spacy.lang.ru import Russian
from sklearn.model_selection import train_test_split
import random
from thinc.api import Config
import re

class ModelSpacy:
    def __init__(self, model_dir = None, use_gpu = True):
        '''
        Создание модели NER и подключение GPU при необходимости

        :param model_dir: Путь до существующей модели (str);
        :param use_gpu: Использовать GPU или нет (bool);
        :return: Возвращаем модель
        '''
        
        if use_gpu:
            spacy.prefer_gpu()
            print('Use GPU: ', spacy.prefer_gpu())

        self.is_model = (model_dir!=None)
        print('Models: ', self.is_model)

        if self.is_model:
            self.model = spacy.load(model_dir)
        else:
            '''
            self.model = spacy.load('ru_core_news_lg')
            pipe_exceptions = ["ner"]
            unaffected_pipes = [pipe for pipe in self.model.pipe_names if pipe not in pipe_exceptions]
            self.ner = self.model.get_pipe('ner')
            self.model.disable_pipes(*unaffected_pipes)
            '''

            config = Config().from_disk('Config\\config.cfg')
            self.model = spacy.blank("ru", config=config)
            self.ner = self.model.get_pipe('ner')


            print("Created 'ru' model")

            #self.model.add_pipe("transformer")

            #self.model.config.to_disk("Config\\config.cfg")

            print(f'Pipline: {self.model.pipe_names}')
            print(self.model.analyze_pipes(pretty=True))


    def fit(self, labels, data_train, data_val = None, epoch = 1 , batch_size = 4, drop = 0.2):
        '''
        Обучение модели

        :param labels: Список меток;
        :param data_train: Тренировочные данные;
        :param data_val: Валидационные данные;
        :param epoch: Количество эпох;
        :param batch_size: Количество батчей;
        :param drop: Исключение части данных при обучении;
        :return: Возвращаем обученную модель
        '''
        if (labels) and (not self.is_model):
            for label in labels:
                self.ner.add_label(label)
        else:
            print('WARNING - Need a list of labels or model')

        if self.is_model:
            optimizer = self.model.create_optimizer()
        else:
            optimizer = self.model.begin_training()


        best_score = 0

        ''' Отключаем ненужные компоненты '''
        pipe_exceptions = ["tok2vec","ner"]
        unaffected_pipes = [pipe for pipe in self.model.pipe_names if pipe not in pipe_exceptions]
        with self.model.disable_pipes(*unaffected_pipes):
            for epoch in range(1, epoch+1):
                random.shuffle(data_train)
                loss = {}
                for batch in spacy.util.minibatch(data_train, size=batch_size):
                    example = None
                    for text, annotations in batch:
                        doc = self.model.make_doc(text)
                        example = Example.from_dict(doc, annotations)

                    self.model.update([example], sgd=optimizer, losses=loss, drop=drop)
                
                loss_train = self.metrics(data_train)
                
                if data_val != None:
                    loss_val = self.metrics(data_val)

                    if loss_val>best_score:
                        self.model.to_disk('model_best')
                        best_score = loss_val
                else:
                    if loss_train>best_score:
                        self.model.to_disk('model_best')
                        best_score = loss_train

                    loss_val = 0

                

                print(loss)
                print('Epoch: ',epoch, ' losses_train: ', loss.get('ner'), 'train_accur: ', loss_train, 'val_accur: ', loss_val)




    def metrics(self, data):
        '''
        Метрика accuracy, которая возвращает долю правильных вариантов делённую на общее количество

        :param data: Список словарей с текстом и необходимым промежутком;
        :return: True / Pred
        '''

        true, pred = self.create_true_pred_arr(data)

        true_pred = 0
        for i in range(len(true)):
            if true[i] == pred[i]:
                true_pred+=1
        return true_pred/len(true)


    def create_true_pred_arr(self, data):
        '''
        Создание двух списков: предсказание модели и "правдивого"

        :param data: Список словарей с текстом и необходимым промежутком;
        :return: Возвращает два списка: с предсказанным срезом и правильным срезом;

        '''
        data_true = []
        data_pred = []
        for i in range(len(data)):
            if data[i][1].get('entities'):
                label = data[i][1].get('entities')[0][2]
                left = data[i][1].get('entities')[0][0]
                right = data[i][1].get('entities')[0][1]
                text = data[i][0]
                data_true.append(text[left:right])
            else:
                label = ''
                data_true.append('')

            doc = self.model(data[i][0])
            if list(doc.ents):
                ents = [ent.text for ent in doc.ents if ent.label_ == label]
                if ents:
                    data_pred.append(str(max(ents, key=len)))
                else:
                    data_pred.append('')
            else:
                data_pred.append('')

        return data_true, data_pred


    def predict(self, data):
        '''
        Предсказание необходимого интервала в тексте

        :param data: Данные в формате [(Text, label), (Text, label), (Text, label) ...];
        :return: Список с промежутками, где есть необходимый кусочек текста. Пример [(1,9), (121,310) ...]
        '''
        labels_ner = {'обеспечение исполнения контракта': 'EC', 'обеспечение гарантийных обязательств': 'PWO'}

        pred = []
        for txt, label in data:
            doc = self.model(txt)
            if list(doc.ents):
                ents = [ent.text for ent in doc.ents if ent.label_ == labels_ner.get(label)]
                if ents:
                    slice_text = str(max(ents, key=len))
                    left = txt.find(slice_text)
                    right = left + len(slice_text)
                    pred.append((left, right))
                else:
                    pred.append(None)
            else:
                pred.append(None)

        return pred

if __name__ == "__main__":
    print("Это модуль модели NER")