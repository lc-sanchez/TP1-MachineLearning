import pandas as pd
from feature_engine.transformation import YeoJohnsonTransformer
from feature_engine.outliers import Winsorizer
import warnings

warnings.filterwarnings('ignore')


# <p>Se decidió realizar una prueba con algunos datos ingresados, a ver si se podrían evaluar transacciones de manera
# individual, simulando un flujo de procesamiento de un control de transacción verdadero, como no tenemos acceso a
# datos de una base de datos, simplemente se estableceran por default ciertos datos y otros si se permitirán ingresar
# para prueba</p> <p>[category, amt, state, city_pop, trans_hora, tras_mes, trans_dia, delay_entre_trans,
# edad_usuario, dif_lat_comprador_merch, dif_long_comprador_merch, dif_lat_prev_merch, dif_long_prev_merch,
# gender_F]</p> <p>Se decidio que columnas como state, city_pop, delay_entre_trans, dif_lat_comprador_merch,
# dif_long_comprador_merch, dif_lat_prev_march,dif_long_prev_march serán por default 0</p>
# 
# <p>Los datos que si podrán ingresar son [category, amt, trans_hora, trans_mes, trans_dia, edad_usuario, gender_f]</p>


# De los datos que pueden recibir, los numericos serian amt, trans_hora, trans_mes, edad_usuario
# Igualmente, por analisis anteriores, sabemos que trans_hora y trans_mes no poseen outliers

class Procesador:
    def __init__(self, ruta):
        self.df = pd.read_pickle(ruta)

    def winsorizar(self, data):
        variables_out = ['amt', 'edad_usuario', 'city_pop', 'delay_entre_trans']
        capper = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=variables_out)
        capper.fit(data)
        data = capper.transform(data)
        return data

    # De los datos que pueden recibir, los categoricos serían category, trans_dia
    def codOneHotCasero(self, data):
        generos = {'M': 0, 'F': 1}
        # genero codificado por oneHotEncoderCasero
        data['gender'] = data['gender'].tolist()
        for valor in data['gender']:
            if pd.notnull(valor):  # agregado para valores null (el valor null de la fila agregada)
                data['gender'] = data['gender'].replace({valor: generos[valor]})
        return data

    def codMeanCasero(self, data):
        estados = {'0': 0.0,
                   'ak': 0.016981132075471698,
                   'al': 0.005245309717241211,
                   'ar': 0.00517235840267292,
                   'az': 0.003435468895078923,
                   'ca': 0.005784244144783534,
                   'co': 0.00814121037463977,
                   'ct': 0.002077382498052454,
                   'dc': 0.005812344312205923,
                   'de': 1.0,
                   'fl': 0.006585268683649317,
                   'ga': 0.005601810996431723,
                   'hi': 0.0027354435326299334,
                   'ia': 0.005262182694089309,
                   'id': 0.0019837691614066726,
                   'il': 0.005733838897623232,
                   'in': 0.005148658448150834,
                   'ks': 0.006783788484953905,
                   'ky': 0.005443371378402107,
                   'la': 0.004340567612687813,
                   'ma': 0.005575307045895281,
                   'md': 0.005993967854006796,
                   'me': 0.007209936382914268,
                   'mi': 0.005156649477835074,
                   'mn': 0.006527085829602069,
                   'mo': 0.004973569773194802,
                   'ms': 0.005427600528601095,
                   'mt': 0.00272247745448358,
                   'nc': 0.004923015925460913,
                   'nd': 0.0038549979710536995,
                   'ne': 0.007447864945382324,
                   'nh': 0.00712732544092776,
                   'nj': 0.004796163069544364,
                   'nm': 0.004997866764186018,
                   'nv': 0.008382379168896023,
                   'ny': 0.006646626986503155,
                   'oh': 0.006906196213425129,
                   'ok': 0.0053616287353305085,
                   'or': 0.008012044953487122,
                   'pa': 0.005735970042706677,
                   'ri': 0.02727272727272727,
                   'sc': 0.006611853374443303,
                   'sd': 0.006004543979227524,
                   'tn': 0.007975390224450268,
                   'tx': 0.005048695138918167,
                   'ut': 0.005701467426862323,
                   'va': 0.00676923076923077,
                   'vt': 0.006118286879673691,
                   'wa': 0.0050729232720355105,
                   'wi': 0.005550258785072188,
                   'wv': 0.005682923981160718,
                   'wy': 0.005692992443846393}

        categorias = {'entertainment': 0.0024783542876592847,
                      'food_dining': 0.0016509769191239982,
                      'gas_transport': 0.004693944204346076,
                      'grocery_net': 0.0029481650972454456,
                      'grocery_pos': 0.014097607531665023,
                      'health_fitness': 0.0015486905995645036,
                      'home': 0.0016082524468992406,
                      'kids_pets': 0.0021143893484319018,
                      'misc_net': 0.014457945549638947,
                      'misc_pos': 0.003138534931893792,
                      'personal_care': 0.002424028735758831,
                      'shopping_net': 0.017561485703740914,
                      'shopping_pos': 0.007225383982446517,
                      'travel': 0.0028637025699261858}

        dias = {'Friday': 0.007086003992854892,
                'Monday': 0.004648382504463549,
                'Saturday': 0.006105783824400245,
                'Sunday': 0.004852761005511236,
                'Thursday': 0.006843874121600977,
                'Tuesday': 0.0058354709256242705,
                'Wednesday': 0.006553599902344495}

        # category, state y trans_dia utilizando diccionario
        data['category'] = data['category'].tolist()
        for val in data['category']:
            data['category'] = data['category'].replace({val: categorias[val]})

        data['trans_dia'] = data['trans_dia'].tolist()
        for value in data['trans_dia']:
            data['trans_dia'] = data['trans_dia'].replace({value: dias[value]})

        data['state'] = data['state'].apply(lambda x: x.lower())
        data['state'] = data['state'].tolist()
        for value in data['state']:
            data['state'] = data['state'].replace({value: estados[value]})
        return data

    def codificar(self, data):
        data = self.codOneHotCasero(data)
        data = self.codMeanCasero(data)
        return data

    def normalizar(self, data):
        # Normalizar
        yj_trans = YeoJohnsonTransformer(variables=data.columns.to_list())
        yj_trans.fit(data)
        data = yj_trans.transform(data)
        return data

    # Escalar
    def escalar(self, data):
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        scaler.fit(data)
        data = pd.DataFrame(data=scaler.transform(data), columns=data.columns.tolist())
        return data

    def arreglarcolumna(self, data):
        # data = data.rename(columns={'gender': 'gender_f'})
        data['gender_f'] = data['gender'].fillna(data['gender_f'])
        data = data.drop(columns=['gender'])
        # movemos columna al final
        col = data.pop('gender_f')
        data = pd.concat([data, col], axis=1)
        return data

    def procesamiento(self, data):
        data = self.codificar(data)
        data = self.winsorizar(data)
        data = self.arreglarcolumna(data)
        data = self.normalizar(data)
        data = self.escalar(data)

        return data

    def procesarConsulta(self, dataframe, consultaDataFrame):
        df_unido = dataframe.append(consultaDataFrame, ignore_index=True)
        dataProcesada = self.procesamiento(df_unido)
        fila = dataProcesada.iloc[[len(dataProcesada) - 1]]
        return fila

    def get_df(self):
        return self.df
