# SMS Spam Collection

**Universiteti:** Universiteti i Prishtinës  
**Fakulteti:** Fakulteti i Inxhinierisë Elektrike dhe Kompjuterike  
**Programi Studimor:** Master në Inxhinieri Kompjuterike dhe Softuerike

**Profesorët:**  
- Prof. Dr. Inxh. Lule Ahmedi
- PhD. c Mërgim Hoti

**Studentët:**  
- Vlora Gjoka
- Sadik Zenuni

## Përshkrimi i Projektit

Projekti në lëndën "Machine Learning" synon të trajtojë sfidën e klasifikimit të mesazheve të SMS në dy kategori: 'ham' (mesazhe të dëshiruara) dhe 'spam' (mesazhe të padëshiruara), duke përdorur datasetin "SMS Spam Collection".

### Fazat e Projektit

Projekti është ndarë në tri faza kryesore:

1. **Faza e parë: Përgatitja e Modelit**
   - Në këtë fazë, bëhet përgatitja e të dhënave dhe ndërtimi i modelit fillestar të machine learning për klasifikimin e mesazheve.
   
2. **Faza e dytë: Analiza dhe Evaluimi**
   - Pas trajnimit të modelit, zhvillohet një fazë e analizës së performancës dhe ritrajnimit të modelit bazuar në rezultatet e marra, për të arritur një saktësi më të lartë.
   
3. **Faza e tretë: Aplikimi i Veglave të Machine Learning**
   - Në fazën përfundimtare, aplikohen teknikat e avancuara të machine learning për të optimizuar dhe implementuar modelin në një mjedis të gjallë.

# Faza I: Përgatitja e Modelit

## Përshkrimi i detyrës

Në këtë fazë, ne merremi me përgatitjen e modelit për të klasifikuar mesazhet në dy kategori: 'ham' dhe 'spam'. Kjo përfshin:
- Leximin e të dhënave nga një burim,
- Pastrimin dhe përpunimin e të dhënave,

## Detajet e datasetit

Dataseti i përdorur është "SMS Spam Collection", i disponueshëm në:

[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

Ky dataset përmban të paktën dy atribute:
- **Label**: Etiketa që tregon nëse një mesazh është 'ham' ose 'spam'.
- **Message**: Teksti i mesazhit që do të analizohet dhe klasifikohet.

## Instalimi i librave të nevojshme

Për të ekzekutuar skriptat e këtij projekti të Mësimit të Makinës për datasetin "SMS Spam Collection", është e nevojshme të instalohen disa librarive specifike të Python. Këto librarive përfshijnë:

- pandas
- scikit-learn

## Udhëzime për instalim

Ju mund të instaloni të gjitha varësitë e nevojshme përmes menaxherit të paketave `pip`. Ekzekutoni komandat e mëposhtme në terminalin ose command prompt tuaj për të instaluar secilën librari:

```
pip install pandas
pip install scikit-learn
```
### Struktura e datasetit

```
# Shfaqja e dataseti-it
printo_datasetin("Dataset-i", df)
```
![Dataset](faza1/results/dataset.png)

### Njohuri mbi llojet e të dhënave
```
# Për të fituar njohuri mbi llojet e të dhënave ekzekutojmë këtë komandë:

df.info()
```
![DaInfo_Dataset](faza1/results/info_dataset.png)

### Menaxhimi vlerave null

```
# Komanda për kontrollimin e vlerave null:
df.isnull().sum()
```
- Në dataset-in tonë nuk ka kolona me vlera null.

![Dataset](faza1/results/null_values.png)

### Menaxhimi i duplikateve:
```
# Komanda për kërkimin e duplikateve dhe shfaqja e rezultatit
print("Duplikatet: " + str(df.duplicated().sum()))
```
- Në dataset-in tonë i janë gjetur disa duplikate:
![Dataset](faza1/results/duplicates.png)

- Fshirja e duplikateve
![Dataset](faza1/results/delete_duplicates.png)

### Mostrimi i të dhënave

#### Përpara Mostrimit

Fillimisht, analizuam shpërndarjen origjinale të klasave në datasetin tonë, i cili përfshin mesazhe të klasifikuara si 'ham' (dëshirueshme) dhe 'spam' (të padëshirueshme). Ky hap është thelbësor për të vlerësuar nevojën për mostrim.

![Shpërndarja e Klasave Para Mostrimit](faza1/results/before_moster.png)

#### Procesi i Mostrimit

Për të balancuar datasetin, ne kemi aplikuar *upsampling* në klasën me përfaqësim më të ulët ('spam'). Kjo përfshin zgjedhjen e rastësishme të rreshtave nga klasa e pakicës, me zëvendësim, derisa numri i rreshtave të saj të arrijë numrin e rreshtave në klasën e shumicës ('ham').

#### Pas Mostrimit

Pas përfundimit të procesit të mostrimit, ne kemi krijuar vizualizime të reja për të treguar shpërndarjen e re të balancuar të klasave. Kjo na lejon të vërtetojmë që dataseti tani është më i balancuar dhe i përshtatshëm për trajnimin e modeleve të mësimit të makinës.

![Shpërndarja e Klasave Pas Mostrimit](faza1/results/after_moster.png)

### Agregimi i të dhënave

Në analizën tonë të datasetit "SMS Spam Collection", kemi përdorur agregime për të nxjerrë në pah disa statistika kyçe që ndihmojnë në kuptimin më të mirë të natyrës së të dhënave. Këto përfshijnë gjatësinë e mesazheve dhe shpërndarjen e klasave të mesazheve.

Statistikat e mëposhtme janë nxjerrë nga të dhënat:

- **Gjatësia Mesatare e Mesazheve**: Gjatësia mesatare e të gjithë mesazheve në dataset.
- **Gjatësia Maksimale dhe Minimale e Mesazheve**: Tregon gjatësinë maksimale dhe minimale të mesazheve që janë regjistruar.
- **Numri i Mesazheve 'Spam' dhe 'Ham'**: Tregon se sa mesazhe janë klasifikuar si spam dhe sa si ham.
- **Gjatësia Mesatare e Mesazheve sipas Kategorisë**: Mesatarja e gjatësisë së mesazheve për secilën kategori, që jep një ide mbi karakteristikat e mesazheve spam dhe ham.

Këto të dhëna agreguese na ndihmojnë të përgatisim dhe rregullojmë më mirë modelet tona të mësimit të makinës për përmirësimin e saktësisë së klasifikimit të mesazheve.

![Agregimi i te dhenave](faza1/results/agregation.png)

###Menaxhimi i Outliers

Në datasetin "SMS Spam Collection", ne kemi analizuar dhe trajtuar outliers për të përmirësuar cilësinë e të dhënave për mësimin e makinës. Outliers mund të ndikojnë ndjeshëm në modelin përfundimtar dhe mund të çojnë në përfundime të pasakta.

#### Metodat e Identifikimit

Outliers u identifikuan duke përdorur dy metoda kryesore:

- **Boxplots**: U përdorën për një analizë vizuale, duke na lejuar të shohim shpejt dhe lehtësisht ndonjë vlerë ekstreme në gjatësinë e mesazheve.
- **IQR Score**: Ne kemi përcaktuar vlerat ekstreme duke përdorur rangun ndërkartil të të dhënave, që është më pak i ndjeshëm ndaj outliers të ekstremeve.
- Paraqitja e Outliers

![Outliers](faza1/results/with_outliers.png)

#### Trajtimi i Outliers
![Outliers](faza1/results/outliers.png)

- Pastrimi i Outliers

![Without Outliers](faza1/results/withou_outliers.png)


### Aplikimi i SMOTE
```
# Inicializimi i SMOTE
sm = SMOTE(random_state=42)

# Aplikimi i SMOTE në të dhënat e trajnimit
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Llogaritja e shpërndarjes së klasave para dhe pas SMOTE
class_distribution_before = Counter(y_train)
class_distribution_after = Counter(y_train_res)

# Vizualizimi
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

ax[0].bar(class_distribution_before.keys(), class_distribution_before.values(), color=['blue', 'green'])
ax[0].set_title('Shpërndarja e Klasave Para SMOTE')
ax[0].set_xlabel('Klasa')
ax[0].set_ylabel('Numri i Shembujve')
ax[0].set_xticks(list(class_distribution_before.keys()))
ax[0].set_xticklabels(['Klasa 0', 'Klasa 1'])

ax[1].bar(class_distribution_after.keys(), class_distribution_after.values(), color=['blue', 'green'])
ax[1].set_title('Shpërndarja e Klasave Pas SMOTE')
ax[1].set_xlabel('Klasa')
ax[1].set_ylabel('Numri i Shembujve')
ax[1].set_xticks(list(class_distribution_after.keys()))
ax[1].set_xticklabels(['Klasa 0', 'Klasa 1'])

plt.tight_layout()
plt.show()
```
![SMOTE](faza1/results/smote.png)

# Faza 2: Trajnimi i Modelit

## Trajnimi dhe Testimi i të Dhënave

Në këtë fazë, ne ndajmë të dhënat në setin e trajnimit dhe testimit dhe trajnojmë një model të mësimit të makinës për të bërë parashikime bazuar në përmbajtjen e mesazhit.

## Testimi i Modelit

Për të vlerësuar performancën e modeleve të klasifikimit, përdorim teknika të ndryshme të testimit si:

- **Ndarja e të dhënave në setin e trajnimit dhe testimit**: Rreth 70-80% të të dhënave përdoren për trajnim, ndërsa pjesa tjetër për testim.
- **Kryqëzimi i Validimit (Cross-Validation)**: Vlerëson aftësinë e generalizimit të modelit në sete të dhënash të padukshme.
- **Metrikat e Performancës**: Përfshinë saktësinë, matricën e konfuzionit, precision, recall, dhe F1 score.

```python
# Ndajmë të dhënat në trajnues dhe testim
X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['Label'], test_size=0.2, random_state=42)

# Përdorimi i TF-IDF Vectorizer për të kthyer tekstet në një format të përpunueshëm numerik
vectorizer = TfidfVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)

# Trajnojmë modelin duke përdorur Naive Bayes
model = MultinomialNB()
model.fit(X_train_transformed, y_train)

# Parashikimi dhe vlerësimi i modelit
predictions = model.predict(X_test_transformed)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))
```
![Train_Test_Data](faza1/results/train_test.png)

## Algoritmet e Klasifikimit

Për datasetin "SMS Spam Collection", eksplorojmë disa algoritme të klasifikimit:

### Naive Bayes
- **Përshtatshëm për**: Datasete të vogla me trajnim të shpejtë.
- **Arsyeja e Përdorimit**: Efikas për tekstin, ofron performancë të lartë në datasete me dimension të lartë.

### Support Vector Machine (SVM)
- **Përshtatshëm për**: Ndarje të qartë mes klasave, hapësirë të madhe të karakteristikave.
- **Arsyeja e Përdorimit**: Efektiv në raste me ndarje të qartë mes klasave.

### Random Forest
- **Përshtatshëm për**: Zvogëlimin e overfitting dhe menaxhimin e të dhënave jo-lineare.
- **Arsyeja e Përdorimit**: Metodë e qëndrueshme, përdor një ansambël pemësh vendimmarrëse.

### Logistic Regression
- **Përshtatshëm për**: Modele probabilitetike që tregojnë gjasat e përkatësisë në një klasë.
- **Arsyeja e Përdorimit**: Intuitive dhe shpesh përdoret për klasifikimin binar.


# Kontributi
Vlora Gjoka

Sadik Zenuni
