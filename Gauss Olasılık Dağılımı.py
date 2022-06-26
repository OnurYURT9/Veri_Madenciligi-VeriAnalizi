# -*- coding: utf-8 -*-
"""
Created on Tue May 24 08:37:11 2022

@author: ONUR
"""

'''Gauss Olasılık Dağılımı

Bir önceki kısmımızda elde ettiğimiz sonuçları incelerseniz sıfırıncı sınıfa ait olan x1'in ortalama değerinin
2.74 küsür olduğunu görebilirsiniz. Burada gördüğünüz x1 gibi belirli gerçek değeri gözlemleme olasılığını hesaplamak
işlemsel olarak da zordur. Bunu yapmanın en kolay yolu x1 değerlerinin çan eğrisi veya gauss dağılımına sahip olduğunu 
varsaymaktır. E zaten gauss dağılımını sadece ortalama ve standart sapma kullanarak özetleyebilirdik. Bunların üzerine biraz
daha matematiksel işlem ekleyerke verilen bir değerin olasılığını tahmin edebiliriz. İşte bu matematik işlemine gauss olasılık
dağılımı fonksiyonu denir.

f(x) = (1 / sqrt(2 * PI) * sigma) * exp(-((x-mean)^2 / (2 * sigma^2)))

Formüle dikkat ederseniz sigmanın x için standart sapma olduğu yerde ortalamayı standart pi değerimiz olduğu kaynaklarda ifade
edilmekte.
'''

from math import sqrt
from math import pi
from math import exp
def calculate_probability(x, mean, stdev):
    exponent =  exp(-((x-mean)**2 / (2 * stdev**2)))
    return (1/(sqrt(2 * pi) * stdev)) * exponent
print(calculate_probability(1.0, 1.0, 1.0))
print(calculate_probability(2.0, 1.0, 1.0))
print(calculate_probability(0.0, 1.0, 1.0))

'''
Örneğimizi çalıştırdığımızda  gördüğümüz gibi bazı giriş değişkenlerinin olasılıkları çalıştırılmaktadır. Giriş değeri 1 ortalama 
değer 1 ve standar sapma 1 olduğunda girdi değeri için sahip olduğumuz değerin en olası değer olan çan eğrisinin üstü olduğunu
görebilirsiniz. Hesaplattırdığımız diğer bir sonuç için ise ortalama ve standart sapmaları aynı tuttuğumuzda giriş değişkeni 
değerlerini yani 0 ve 2'nin her ikisinin de 0.24 ortalamaya sahip olup çan eğrisi yapısında olup birbirlerine eşit farklılıklarda 
olduğunu  görürsünüz.


5.ADIM
Artık yeni veriler için olasılıkları hesaplamada eğitim verilerimizden hesaplanan istatistiklerimizi kullanabiliriz. Bunu yaparken
olasılıkların her sınıf için ayrı ayrı hesaplanacağını unutmayın. Bu işlem kısaca şöyle gerçekleştirilebilir:
    Üzerinde işlem yaptığımız yeni bir veri parçasında o parçanın önce 1. sınıfa ait olma olasılığı hesaplanır ardından 2. sınıfa
    ait olma olasılığı hesaplanır. Bu kaç tane sınıf varsa böyle devam eder ve her yeni veri parçası için tekrarlanır.
Bir veri parçasının bir sınıfa ait olma olasılığı şu şekildedir:
    P(class|data) = P(X|class) * P(class)

İki girdi değişkenine sahip olduğumuz önceki örneğimiz için her bir satırın 0. sınıfa ait olma olasılığını şöyle hesaplarız:
    P(class=0|X1,X2) = P(X1|class=0) * P(X2|class=0) * P(class=0)
    
Bir önceki adımda gauss olasılık yoğunluk fonksiyonu x1 gibi gerçel bir değerin olasılığının hesaplanmasında nasıl kullanıldığını
gördük ve aynı zamanda istatistik değerlerinin de nasıl kullanıldığını gördük.Şimdi bunları class olasılıklarını hesapla diye
bir fonksiyonla birbirlerine bağlayacağız. Bunu için şu kodları kullancağız:
 
  def calculate_class_probabilities(summaries, row):
        total_rows = sum([summaries[label][0][2] for label in summaries])
        probabilities = dict()
        for class_value, class_summaries in sumaries.items():
            probabilites[class_value] = summaries[class_value][0][2] / float(total_rows)
            for i in range(len(class_summaries)):
                mean, stdev, count = class_summaries[i]
                probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
            return probabilites

Eğer buradaki fonksiyona dikkat ederseniz girdi argümanı olarak verilerin özeti ve işlenecek olan yeni bir veri satırını alır.
Devamında özet istatistiklerde saklanan sayılardan toplam eğitim kaybı sayısını hesapalar (total_rows kullaranak bunu hesaplıyor).
Ayrıca bu değer eğitim verilerindeki tüm satırların belirli bir sınıfa sahip satırların oranı olarak belirli bir sınıfın veya
P(class) olasılığının(probabilities) hespalanmasında kullanılmaktadır. Sonrasında gauss olasılık yoğunluk fonksiyonu ve o sütunun
ve bulunduğu o sınıfın istatistiklerini kullanarak 
=> bu kısımda [for i in range(len(class_summaries)):
    mean, stdev, count = class_summaries[i]] satırdaki her girdi değeri için olasılıkları hesaplar devamında da 
    

  hesaplanan her yeni olasılık birbirleriyle çarpılır.Şu kısım =>
  probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    
    
Ve bu işlem veri setindeki her sınıf için tekrarlanır.Bize en son geri döndürdüğü değer ise sözlük yapısı ile tanımladığı
olasılıklardır. => probabilities = dict()

Bu döndürülen sözlük yapısı her sınıf için bir giriş değeri içeren olasılık sözlüğüdür.

'''

from math import sqrt
from math import pi
from math import exp
 
# Veri kümesini sınıf değerlerine göre böl, bir sözlük döndürür
def separate_by_class(dataset):
	separated = dict()
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = vector[-1]
		if (class_value not in separated):
			separated[class_value] = list()
		separated[class_value].append(vector)
	return separated
 
# Bir sayı listesinin ortalamasını hesaplayın
def mean(numbers):
	return sum(numbers)/float(len(numbers))
 
# Bir sayı listesinin standart sapmasını hesaplayın
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)
 
# Bir veri kümesindeki her sütun için (mean)ortalamayı, (standart sapma) stdev'i ve (count) sayımı hesaplayın
def summarize_dataset(dataset):
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	del(summaries[-1])
	return summaries
 
# Veri kümesini sınıfa göre ayırın ve ardından her satır için istatistikleri hesaplayın
def summarize_by_class(dataset):
	separated = separate_by_class(dataset)
	summaries = dict()
	for class_value, rows in separated.items():
		summaries[class_value] = summarize_dataset(rows)
	return summaries
 
# x için Gauss olasılık dağılım fonksiyonunu hesaplayın
def calculate_probability(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent
 
# Belirli bir satır için her sınıfı tahmin etme olasılıklarını hesaplayın
def calculate_class_probabilities(summaries, row):
	total_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
	for class_value, class_summaries in summaries.items():
		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
		for i in range(len(class_summaries)):
			mean, stdev, _ = class_summaries[i]
			probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
	return probabilities
 
# Test hesaplama sınıfı olasılıkları
dataset = [[3.393533211,2.331273381,0],
	[3.110073483,1.781539638,0],
	[1.343808831,3.368360954,0],
	[3.582294042,4.67917911,0],
	[2.280362439,2.866990263,0],
	[7.423436942,4.696522875,1],
	[5.745051997,3.533989803,1],
	[9.172168622,2.511101045,1],
	[7.792783481,3.424088941,1],
	[7.939820817,0.791637231,1]]
summaries = summarize_by_class(dataset)
probabilities = calculate_class_probabilities(summaries, dataset[0])
print(probabilities)

#0. sınıfa ait olma olasılığı 0.050324...
#1. sınıfa ait olma olasılığı 0.000115..













