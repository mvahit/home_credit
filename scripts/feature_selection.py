# EKSİK DEĞER, AYKIRI DEGER, FEATURE SCALING:
# DOĞRUSAL MODELLERDE, SVM, YSA, KNN YA DA UZAKLIK TEMELLIK YONTEMLERDE ONEMLIDIR.

# AĞAÇ YÖNTEMLERİNDE ÖNEMİ DÜZEYLERİ ÇOK AZDIR.

# Filter Methods (Statistical methods: korelasyon, ki-kare)
# Wrapper Methods (backward selection, forward selection, stepwise)
# Embeded (Tree Based Methods, Ridge, Lasso)

# Tree Based Methods
# korelasyon, ki-kare

# TODO TREE BASED SELECTION

# TODO: tum değişkenleri, sayısal değişkenleri, kategorik değişkenleri (iki sınıflı ya da çok sınıflı),
#  yeni türetilen değişkenlerin isimlerini ayrı listelerde tut.

all_cols = []  # target burada olmamalı
num_cols = [col for col in df.columns if df[col].dtypes != 'O']
cat_cols = []
new_cols = []
target = []

# TODO: random forests, lightgbm, xgboost ve catboost modelleri geliştir.
#  Bu modellere orta şekerli hiperparametre optimizasyonu yap. Final modelleri kur.
#  Bu modellerin her birisine feature importance sor. Gelen feature importance'ların hepsini bir df'te topla.
#  Bu df'in sütunları aşağıdaki şekilde olsun:

# model_name feature_name feature_importance

# TODO: oluşacak df'i analiz et. Grupby ile importance'in ortalamasını alıp, değişken önemlerini küçükten büyüğe sırala.
#  En önemli değişkenleri bul. Sıfırdan küçük olan importance'a sahip değişkenleri sil.
#  Nihayi olarak karar verdiğin değişkenlerin adını aşağıdaki şekilde sakla:

features_based_trees = []

# TODO: Önemli not. Yukarıdaki işlemler neticesinde catboost'un sonuçlarına özellikle odaklanıp
#  kategorik değişkenlerin incelenmesi gerekmektedir.
#  Çalışmanın başında tutulmuş olan cat_cols listesini kullanarak
#  sadece categorik değişkenler için hangi ağacın nasıl bir önem düzeyi verdiğini inceleyiniz
#  ve diğer algoritmalarca önemsiz catboost tarafından önemli olan değerlendirilen değişkenleri bulunuz
#  ve aşağıdaki şekilde kaydediniz:

features_catboost_cat = []

# TODO: features_based_trees listesinde yer ALMAYIP catboost_cat_imp listesinde YER ALAN değişkenleri bulunuz
#  ve bu değişkenleri features_based_trees listesine ekleyiniz.


# TODO STATISTICAL SELECTION

# TODO bağımsız değişkenlerin birbiri arasındaki korelasyonlarına bakıp birbiri ile
#  yüzde 75 üzeri korelasyonlu olan değişkenler arasından 1 tane değişkeni rastgele seçiniz
#  ve değişkenlerin isimlerini aşağıdaki gibi kaydediniz:
#  elenen değişkenlerin isimlerini de aşağıdaki gibi kaydediniz:

features_based_correlation = []
features_dropped_based_correlation = []


# TODO: features_based_trees listesinde olup aynı anda features_dropped_based_correlation listesinde olan feature'lara
#  odaklanarak inceleme yapınız ve gerekli gördüğünüz değişkenleri features_based_trees listesinden siliniz ya da
#  drop listesinden agaç listesine taşıyınız

# TODO: veri setindeki kategorik değişkenler ile bağımlı değişken arasında chi-squared testi uygulayınız
#  ve bu test sonucuna göre target ile dependency'si bulunan değişkenleri aşağıdaki şekilde saklayınız:

cat_cols_chi = []

# TODO: yukarıdan gelecek olan değişkenler ile features_based_trees listelerini karşılaştırınız. Durumu analiz ediniz.
#  cat_cols_chi listesinde olup features_based_trees listesinde olmayan değişkenleri eklemeyi değerlendiriniz.
#  ya da cat_cols_chi'de olmayıp features_based_trees'de olan değişkenkeri çıkarmayı değerlendiriniz.
#  Değerlendirmekten kastım sizin yorumunuza kalmış.


# TODO: netice olarak en sonda aşağıdaki isimlendirme ile seçilmis feature'ları kaydediniz:


features_selected = []

# TODO: seçilmiş feature'lar ile model tuning yaparak lightgbm için hiperparametre optimizasyonu yapınız.
# TODO: yeni hiperparametrelerle final modeli oluşturunuz.