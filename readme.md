dataLoader.py: Projenin veri yönetimini ve veri setinin bölünmesini sağlayan modüldür. Diskten ham verileri (IMDB ve Tweets) okur, temizleme standardizasyon işlemlerini yapar; ardından hedef etiketleri tamsayı indekslerine haritalayarak veri setini %64 Eğitim (Train), %16 Geçerleme (Validation) ve % 20 Test kümesi olarak katmanlı örnekleme yöntemiyle üç bağımsız parçaya böler.

featureExtractor.py: Metinsel verileri TF-IDF yöntemiyle sayısal vektör haline getiren modüldür. fit_transform_train() ve transform_unseen() fonksiyonları sayesinde eğitim kümesi dışındaki verilerden bilgi sızmasını engeller.

dataPreparer.py: dataLoader.py ve featureExtractor.py modüllerini birleştirerek IMDB ve Tweets veri setleri için hazırlık sürecini yürüten modüldür. Her veri seti için ham metinleri yükler, katmanlı örnekleme ile train, validation ve test kümelerine ayırır. TF-IDF vektörleştiricisini sadece eğitim kümesinde fit ederek doğrulama ve test kümelerine sızıntısız şekilde uygular. Eğitim kümesindeki sınıf dağılımını analiz ederek dengesiz sınıflar için ters orantılı olacak şekilde ağırlık hesaplaması yapar.

MLP: PyTorch nn.Module kütüphanesini kullanarak MLP mimarisini tanımlayan modüldür. Gizli katman sayısı ve boyutu, aktivasyon fonksiyonu (ReLU/Tanh/LeakyReLU), Batch Normalizasyon kullanımı ve Dropout oranı parametre olarak verilir, bu sayede hiper parametre optimizasyon sürecinde aynı sınıf farklı mimari konfigürasyonları ile yeniden kullanılabilir. Son katman aktivasyonsuz bırakılarak CrossEntropyLoss ile uyumlu ham çıktı üretilir.

trainer.py: Modelin eğitim ve doğrulama süreçlerini yürüten MLPTrainer sınıfını içeren modüldür.. Model parametrelerini Adam algoritması ve CrossEntropyLoss fonksiyonu ile günceller ve Early Stopping mekanizmasına sahiptir.



evaluator.py: Modelin doğrulama ve test kümeleri üzerindeki başarısını ölçen Evaluator sınıfını barındırır. Tahminleri üretir ve Doğruluk (Accuracy), Macro F1-Skoru, Ortalama Kare Hata (MSE) ve Karmaşıklık Matrisini (Confusion Matrix) mlp_project_results.txt dosyasına yazılacak format haline getirir. 

main.py: İki adımdan oluşan hiper parametre aramasını yapan ve tüm sonuçları bir text dosyasına yazan ana modüldür. Birinci adımda en uygun H1, H2, Activation Function, Batch normalization ve Dropout Rate hiper parametreleri ni seçer. İkinci adımda ise bu dört hiper parametrenin oluşturduğu en iyi mimari üzerinde Batch size ve Learning Rate hiper parametreleriyle fine tuning gerçekleştirir. Son olarak tüm süreçleri ve nihai test sonuçlarını konsola ve mlp_project_results.txt dosyasına kaydeder. 
