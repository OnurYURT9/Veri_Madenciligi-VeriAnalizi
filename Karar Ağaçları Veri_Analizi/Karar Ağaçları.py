# -*- coding: utf-8 -*-
"""
Created on Tue May 31 08:36:33 2022

@author: ONUR
"""
#Veri setinde ilk beş sütun giriş son sütun çıkış. Son sütun 0-1 diye ikiye ayrılmış. Yani ikili bir sınıflandırma işlemi içeriyor.
# Bu yapıyı karar ağacında kullanıcaz.
'''
Karar ağaçları son derece popüler ve güçlü bir tahmin yöntemidir. Karar ağacından elde edilen modelin hem kullanıcılar hem de alanında 
uzman uygulayıcılar tarafından anlaşılması çok kolay olduğu için popülerdir. Karar ağacında belli bir tahminin neden yapıldığı açıklana-
bilir bu da kullanımındaki tercih sebeplerinden bir tanesidir. Karar ağacının tercih edilmesinin diğer bir sebebi de bölünme mekanizmalarının
kuralları değiştiriliebilir en güncel uygulamalarından bir tanesi bölünme kurallarıyla birliktelik kuralının entegre edilebilirliğidir. 

Karar ağaçları ayrıca bagging dediğimiz torbalama rastgele orman (random forest) ve gradyan artırma gibi daha gelişmiş derleme yöntemleri 
için temel sağlar. Bu alanda bilinen sınıflandırma ve regresyon ağaçları CART (Classification and Regression Tree) sınıflandırma ve regresyon 
öngörüsü modelleme problemlerinde kullanılabilir. Bu model ikili bir ağaçla temsil edilebilir.  Yani kısaca her bir düğüm 0 ya da 1 değerlerini
temsil edebilir. Bunu temsil etmez ise 2 alt düğüm sunabilir. Bir düğüm değişkenin sayısal olduğubir varsayılarak tek bir girdi değişkeni olan 
x'i ve bu değişken üzerindeki bir bölünme noktasını temsil eder.

Ağacın yaprak düğümleri ki bunlar terminal düğümleri olarak da adlandırılır. Bir tahmin yapmak için kullanılan çıktı değişkenini yani y
değerini içerir. Bu yapıda ağaç bir kez oluşturulduktan sonra son bir tahmin yapılana kadar bölmelerle birlikte her dalı izleyen yeni bir veri 
satırıyla gezinilebilir. Sizin de hatırlayacağınız gibi ikili bir karar ağacı oluşturmak aslında girdi alanını bölme işlemidir. Bu bölme 
işleminde greedy yani aç gözlü denilen bir yaklaşım kullanılır. Temeli recursive (özyinelemeli) ikili bölme mantığını içerir. Bu yapı tüm 
değerlerin sıralandığı maliyet fonksiyonunu kullanarak farklı dallanma noktalarının denendiği ve test edildiği sayısal bir prosedürdür. Bu 
prosedürde en iyi(düşük) maliyetli bölünme seçilir. Tüm girdi değişkenleri ve olası dallanma noktaları bu açgözlü maliyet fonksiyonuna 
dayanılarak seçilir. 

Regresyon işleminde bölünme noktalarını seçmek için en aza indirilen maliyet işlemi hesabın yapıldığı alandaki (dörtgen alanı denir)
tüm eğitim örneklerindeki toplam kare hatasına göre seçim işlemi yapılır. Sınıflandırmada ise ağaçta oluşturulan düğümlerin geçerliliğinin 
(saflığının) bir göstergesi olan Gini fonksiyonu kullanılır. Burada düğüm saflığı her bir düğüme atanan eğitim verilerinin ne kadar karışık 
olduğunu ifade eder. Dallanma işlemi (bölmede) düğümler minimum sayıda eğitim örneği içerene veya maksimum ağaç derinliğine ulaşılana kadar
devam eder.

Ağaç oluşturma işlemi 4 kısımdan oluşmaktadır. Bunlar Gini indexi, bölme(dallanma) işlemi, ağacın oluşturulması ve son olarak tahmin yapmadır.

Gini Index'i: Veri kümesindeki bölünmeleri değerlendirmek için kullanılan maliyet işleminin adıdır. Veri kümesindeki bir bölünme bir nitelik 
için bir girdi değerini içerir ve ikili sıra grubuna bölmek için kullanılabilir.  Bir Gini puanı bölünmenin yarattığı iki gruptaki sınıfların
ne kadar karışık olduğuna göre bir bölünmenin ne kadar iyi olduğuna dair bir fikir sağlar. Mükemmeler ayrım 0 Gini puanıyla sonuçlanırken;
her gruptan yarı yarıya (50'ye 50) sınıfla sonuçlanan en kötü durum ayrımı ise 0.5 Gini puanı ile sonuçlanır. 
                         
Her grupta iki satır veri olan bir veri grubumuz olduğu düşünülürse 1. gruptaki satırların tümü sıfırıncı sınıfa ve 2. gruptaki satırlar ise 
1. sınıfa aittir diyebiliriz. Bu sebepten bu yapı mükemmel bir bölünmedir. Burada da öncelikle her gruptaki sınıfların oranını hesaplamamız 
gereklidir. Bu oranı şu şekilde hesaplarız:
    orantı = say(sınıf_Degeri) / say(satır)
Bu örneğimizdeki oranları ise hesapladığımızda şunu elde ederiz:
    grup_1_sınıf_0 = 2/2 = 1
    grup_1_sınıf_1 = 0/2 = 0
    grup_2_sınıf_0 = 0/2 = 0
    grup_2_sınıf_1 = 2/2 = 1

Bu işlemin devamında Gini indexi her alt düğüm için şu şekilde hesaplanır:
    gini_index = toplam(oran * (1.0 - orantı))
    gini_index = 1.0 - toplam(oran *  orantı)


Her grup için Gini index'i daha sonra parent denilen ebeveyndeki tüm örneklere göre ve grubun büyüklüğüne göre ağırlıklandırılmalıdır. Bu 
ağırlıklandırmayı bir grup için Gini indexi hesaplamasına şöyle ekleyebiliriz:
    gini_index = (1.0 - toplam(oran *  orantı)) * (grup_boyutu / toplam_ornekler)
    Gini(grup_1) = (1.0 - (1*1+0*0))*(2/4)
    Gini(grup_1) = 0*0.5 =0
    
Mükemmel şekilde ayrıldığı için Gini 0.

    Gini(grup_2) = (1.0 - (0*0+1*1))*(2/4)
    Gini(grup_2) = 0.5*0 =0

Burada elde edilen puanlar daha sonra diğer aday bölünme noktalarıyla karşılaştırılabilecek olan bölünme noktası için kesin bir Gini puanı 
vermek üzere bölünme noktasındaki her bir alt düğüme eklenir. Bu bölünme noktası için Gini indexi 0 olarak hesaplanır. 

Şimdi yazacağımız örneğiizde Gini index adlı bir işlev bulunmakta. bu işlev bir grup listesi ve bilinen sınıf değerleri listesi için Gini 
indexini hesaplar.


'''
def gini_index(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p*p
        gini += (1.0 - score) * (size / n_instances)
    return gini

print (gini_index( [[[1,1], [1,0]] , [[1,1],  [1,0]]], [0,1]))

'''
İkinci olarak yapacağımız işlem bölme işlemi. Bölme işleminde veri kümesindeki bir öznitelikten faydalanılır. Bu işlevi bölünecek bir 
özniteliğin indexi ve bu öznitelik üzerindeki veri satırlarının bölüneceği değer olarak düşünebiliriz. Bu yöntem aslında veri satırlarını
indexlemek için kullanılabilecek hızlı bir çözümdür. Bölme işleminde 3 kısım kullanılmaktadır. Bu kısımalrdan ilki Gini indexidir. Diğeri
veri setini bölme işlemidir. Üçüncüsü ise tüm bölümleri değerlendirme işlemidir. Veri setini  ya da veri kümesini bölme şöyle olur:
    
   Bir veri kümesini bir özniteliğin dizini ve bu öznitelik için bir bölme değeri verilen 2 satır listesine ayırma işlemi ile gerçekleştirilir.
İki grubumuz olduğunda bölünmenin maliyetini değerlendirmek için Gini indexini kullanabiliriz. Şunu unutmayın! Bir veri kümesini bölmek
her satır üzerinde yinelemeyi öz nitelik değerinin bölünmüş değerin altında mı yoksa üstünde mi olduğunu kontrol etmeyi ve onu sırasıyla sol 
veya sağ gruba atamayı içerir. 

'''

def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


'''
Tüm bölümleri değerlendirme:
    Bu işlemi yapabilmek için gereklilik olan Gini işlevi ve test işlevi fonksiyonuna artık sahibiz. Bir veri kümesi verildiğinde aday 
 bölme olarak olarak her bir öznitelik üzerindeki her değeri kontrol etmeli bölmenin maliyetini değerlendirmeli ve yapabileceğiniz en iyi
 olası bölmeyi (ya da dallanmayı) bulmalıyız. En iyi bölünme bulunduğunda bu bölünmeyi karar ağacında bir düğüm olarak kullanabiliriz. 
 Bu yapı esasen kapsamlı ve aç gözlü bir algoritmadır. Verileri ada göre depolayabildiğimizden karar ağacındaki bir düğümü temsil etmede
 dictionary yani sözlük yapısı kullanacağız. En iyi bölmeyi seçip ağaç için yeni bir düğüm olarak kullanırken seçilen özniteliğin indeksini
 bölünecek özniteliğin değerini ve seçilen bölme noktasına göre bölünen iki veri grubunu saklayacağız. Her veri grubu bölme işlemiyle 
 sol ya da sağ gruba atanan satırlardan oluşan kendi başına küçük bir veri kümesidir. Yazacağımız kodlarda buna yönelik olarak getsplit adlı
 bir işlevimiz bulunmaktadır. Her bir öznitelik üzerinde (sınıf değeri hariç) ve ardından öznitelik için her bir değer yinelendiğinde 
 bölmeleri ya da kısımları bölme ya da değerlendirme işlemi yapılır. Sonuç olarak en iyi dallanma değeri kaydedilir ve sonuçlar geri 
 döndürülür(en iyisi).

 
'''

# def get_split(dataset):
#     class_values = list(set(row[-1] for row in dataset))
#     b_index, b_value, b_score, b_groups = 999, 999,999, None
#     for index in range(len(dataset[0])-1):
#         for row in dataset:
#             groups = test_split(index, row[index], dataset)
#             gini = gini_index(groups, class_values)
#             if gini < b_score:
#                 b_index, b_value, b_score, b_groups = index, row[index], gini, groups
#     return {'index':b_index, 'value':b_value, 'groups':b_groups}

def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999,999, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            print('X%d < %.3f Gini =%.3f' %((index+1), row[index],gini))
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}


def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def gini_index(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p*p
        gini += (1.0 - score) * (size / n_instances)
    return gini

dataset = [[2.771244718, 1.784783929,0],
           [1.728571309, 1.169761413,0],
           [3.678319846, 2.81281357,0],
           [3.961043357, 2.61995032,0],
           [2.999208922, 2.209014212,0],
           [7.497545867, 3.162953546,1],
           [9.00220326, 3.339047188,1],
           [7.444542326, 0.476683375,1],
           [10.12493903, 3.234550982,1],
           [6.642287351, 3.319983761,1]]

split = get_split(dataset)
print('Split: [X%d < %.3f]' %((split['index']+1), split['value']))






















