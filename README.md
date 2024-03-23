
# Faza I: Përgatitja e modelit

Ky dokument përmban përshkrimin e fazës së parë të projektimit dhe përgatitjes së modelit për klasifikimin e mesazheve në 'ham' (mesazhe të dëshiruara) dhe 'spam' (mesazhe të padëshiruara), duke përdorur datasetin "SMS Spam Collection".

## Përshkrimi i detyrës

Në këtë fazë, ne merremi me përgatitjen e modelit për të klasifikuar mesazhet në dy kategori: 'ham' dhe 'spam'. Kjo përfshin:
- Leximin e të dhënave nga një burim,
- Pastrimin dhe përpunimin e të dhënave,
- Ndarjën e të dhënave në setin e trajnimit dhe testimit,
- Trajnimin e një modeli të mësimit të makinës për të bërë parashikime bazuar në përmbajtjen e mesazhit.

## Detajet e datasetit

Dataseti i përdorur është "SMS Spam Collection", i disponueshëm në:

[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

Ky dataset përmban të paktën dy atribute:
- **Label**: Etiketa që tregon nëse një mesazh është 'ham' ose 'spam'.
- **Message**: Teksti i mesazhit që do të analizohet dhe klasifikohet.

## Algoritmet e klasifikimit

Për datasetin "SMS Spam Collection", ne mund të përdorim disa algoritme të ndryshme të klasifikimit për të ndarë mesazhet në "spam" dhe "ham". Këto algoritme përfshijnë:

### Naive Bayes
- **Përshtatshëm për**: Kur kemi datasete relativisht të vogla dhe duam një model që është i shpejtë dhe efektiv në kohën e tij të trajnimit.
- **Arsyeja e përdorimit**: Naive Bayes është një zgjedhje popullore për tekstual data si SMS për shkak të thjeshtësisë së tij dhe performancës së mirë me datasete me dimenzione të larta.

### Support Vector Machine (SVM)
- **Përshtatshëm për**: Kur kërkojmë një margin maksimal ndërmjet klasave dhe kemi një hapësirë të madhe karakteristikash.
- **Arsyeja e përdorimit**: SVM mund të jetë efektiv në rastet kur ka një ndarje të qartë mes klasave dhe kur hapësira e karakteristikave është e madhe, që është e zakonshme në analizën e tekstit.

### Random Forest
- **Përshtatshëm për**: Kur duam të zvogëlojmë rrezikun e overfitting dhe të trajtojmë mirë të dhënat jo-lineare.
- **Arsyeja e përdorimit**: Random Forest është një metodë e qëndrueshme që përdor një sërë pemësh vendimmarrëse për të rritur saktësinë e parashikimeve dhe për të menaxhuar mirë varësitë në të dhëna.

### Logistic Regression
- **Përshtatshëm për**: Rastet kur dëshirojmë një model probabilitetik që tregon gjasat e përkatësisë në një klasë.
- **Arsyeja e përdorimit**: Përdoret shpesh për probleme të klasifikimit binar siç është rasti me SMS spam dhe ham, ofron një mënyrë intuitive për të kuptuar rëndësinë e karakteristikave të ndryshme.


## Rezultatet e fazës së parë

Në fund të kësaj faze, pritet të kemi një skicë të qartë të procesit të trajnimit të modelit, duke përfshirë përgatitjen e të dhënave, zgjedhjen e modelit, dhe një vlerësim të parë të performancës së modelit në datasetin e testit.
