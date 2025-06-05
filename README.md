# Modelatge dels patrons de moviment d'esportistes de futbol americà per a predir el tipus d'atac

## Crèdits
 * __Autor__: Lluís Coll Mas
 * __Tutor__: Joaquín Torres-Sospedra
 * __Responsable Docent__: Ismael Benito Altamirano

## Descripció del repositori

Aquest repositori està dividit en dues grans parts. Una amb arxius `python` enfocats més a les tasques prèvies al modelatge i una altra amb notebooks `ipynb` que estan enfocats en tots dos àmbits. Fem una breu descripció de cada apartat. 
 * `/src/images`: carpeta amb algunes de les representacións que es fan servir al treball. 
 * `/src/preprocessat`: carpeta que conté els arxius `python` utilitzats per a fer el preprocessament. 
     * `/build_nested_dataset.py`: genera els arxius `.pt` que ens permetran llegir les dades pel modelatge. 
     * `/combined_data.py`: és el procés de transformació de les característiques dels diferents jocs de dades en un de sol. 
     * `/seleccio_dades.py`: creació i adaptació de les variables en cadascun dels jocs de dades. 
     * `/standardize_tracking.py`: procés d'estandardització per al joc de dades de tracking. 
     * `/tracking_data.py`: selecció del joc de dades de tracking. 
     * `/train_test_creator.py`: creació dels sets d'entrenament i test. 
 * `/notebooks`: carpeta amb els notebooks `ipynb` per a fer part del preprocés, així com certa visualització i també el model.  
    * `/model/DeepSets_100Epochs.ipynb`: és el model que he creat a partir de les dades transformades. 
    * `/DataAnalysis.ipynb`: procés d'anàlisi i visualització de les dades obtingudes en cadascun dels jocs de dades transformats. 

 ## Origen de les dades
 
 En aquest repositori no s'inclouen les dades originals a causa del seu pes. Però s'han extret del següent concurs de Kaggle: [NFL Big Data Bowl 2025](https://www.kaggle.com/competitions/nfl-big-data-bowl-2025/data).

Els arxius `.pt` finals amb què s'ha generat el model tampoc s'han pogut pujar al repositori pel pes. Però sí són accessibles a través del Drive de la UOC en el següent enllaç: [Arxius `.pt` al Drive](https://drive.google.com/drive/folders/1m8bEdQTql4mjFxWP2goJPcC8qOHrWYSN?usp=sharing)
