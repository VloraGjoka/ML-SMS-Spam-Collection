
# Faza I: Përgatitja e Modelit

Ky dokument përmban përshkrimin e fazës së parë të projektimit dhe përgatitjes së modelit për klasifikimin e mesazheve në 'ham' (mesazhe të dëshiruara) dhe 'spam' (mesazhe të padëshiruara), duke përdorur datasetin "SMS Spam Collection".

## Përshkrimi i Detyrës

Në këtë fazë, ne merremi me përgatitjen e modelit për të klasifikuar mesazhet në dy kategori: 'ham' dhe 'spam'. Kjo përfshin:
- Leximin e të dhënave nga një burim,
- Pastrimin dhe përpunimin e të dhënave,
- Ndarjën e të dhënave në setin e trajnimit dhe testimit,
- Trajnimin e një modeli të mësimit të makinës për të bërë parashikime bazuar në përmbajtjen e mesazhit.

## Detajet e Datasetit

Dataseti përdorur është "SMS Spam Collection", i disponueshëm në:

[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

Ky dataset përmban të paktën dy atribute:
- **Label**: Etiketa që tregon nëse një mesazh është 'ham' ose 'spam'.
- **Message**: Teksti i mesazhit që do të analizohet dhe klasifikohet.

## Rezultatet e Fazës së Parë

Në fund të kësaj faze, pritet të kemi një skicë të qartë të procesit të trajnimit të modelit, duke përfshirë përgatitjen e të dhënave, zgjedhjen e modelit, dhe një vlerësim të parë të performancës së modelit në datasetin e testit.
