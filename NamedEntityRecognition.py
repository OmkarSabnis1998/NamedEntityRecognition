# NAMED ENTITY RECOGNITION USING DEEP LEARNING
# BY - OMKAR VIVEK SABNIS - 17/07/2018

from process_data import DataHandler
s = DataHandler("M:/Deep and Machine Learning Projects/Named-Entity-Detection using Deep Learning/Dataset/eng.testa")
print(s.get_data()[0].shape)
from NER_model import NER
m = NER(s)
m.make_and_compile(units=100,dropout=0.0,regul_alpha=0.005)
from keras.models import load_model
m.model = load_model("./first_model")
#m.train(epochs=10)
#m.evaluate()

m.predict_tags("Butter Chicken is so good!")

#m.model.save("./first_model")

