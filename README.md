![analytics](https://img.shields.io/badge/Machine%20Learning-Medicina%20predictiva-green)

# Estimación de niveles de  obesidad basados en habitos alimenticios y  condición fisica
 Se utiliza  varios  algoritmos de aprendizaje automatico para la predicción en al condición de salud en diabetes ( presencia o ausencia de la enfermedad), se selecciona el modelo con mejor metricas de desempeño

# Introducción 👇


<p style='text-align: justify;'>

La obesidad es un trastorno nutricional cuya incidencia ha venido aumentado,
especialmente en los Estados Unidos y Europa. El origen de este incremento está
en los cambios nutricionales, atributos de acondicionamiento fisico y económicos que experimentan las sociedades.
El incremento de la prevalencia de la obesidad, tiene implicaciones significativas sobre las políticas de salud
pública porque los tratamientos de sus comorbilidades —las enfermedades que
acompañan al desorden original o primario— suelen ser costosos y duraderos en el tiempo [(Palechor, F. M., & de la Hoz Manotas, A., 2019)](https://www.sciencedirect.com/science/article/pii/S2352340919306985).
    
[medlinePlus (2021)](https://medlineplus.gov/spanish/ency/patientinstructions/000348.htm) expone que la obesidad es una enfermedad grave y crónica; la cual puede llevar a otros problemas de salud
incluyendo diabetes, enfermedad cardíaca y algunos cánceres. Por ende, las personas con obesidad tienen una mayor probabilidad de sufrir estos problemas de salud:
 
 * Glucosa (azúcar) alta en la sangre o diabetes.
 * Presión arterial alta (hipertensión).
 * Nivel alto de colesterol y triglicéridos en la sangre (dislipidemia o alto nivel de grasas en la sangre).
 * Ataques cardíacos debido a enfermedad cardíaca coronaria, insuficiencia cardíaca y accidente cerebrovascular.
 * Problemas óseos y articulares. Más peso ejerce presión sobre los huesos y articulaciones. Esto puede llevar a osteoartritis,    una enfermedad que causa rigidez y dolor articular.
 * Apnea del sueño o pausas en la respiración durante el sueño. Esto puede causar fatiga o somnolencia diurna, poca atención y    problemas en el trabajo.
 * Cálculos biliares y problemas del hígado.
 * Algunos tipos de cáncer.  

Para el caso de Colombia, en 2015, el 56,4%  de los
adultos tenía sobrepeso, y el 18,7% era obeso [(Instituto Colombiano de Bienestar
Familiar, 2015)](https://www.icbf.gov.co/bienestar/nutricion/encuesta-nacional-situacion-nutricional#ensin3). Según el icbf (2010, p. 9): *«Uno de cada tres colombianos presenta
exceso de peso… Las cifras de exceso de peso aumentaron en los últimos cinco
años en 5.2 puntos porcentuales con respecto al 2010…»*    
   
Apartir en las problematicas de salud pubica, la predicción temprana del riesgo de enfermedad y la medicina personalizada tienen un gran potencial en el futuro de la atención médica, ya que las soluciones de inteligencia artificial (IA) y machine learning (ML) están transformando la forma en que se brinda cuidado de la salud. Las organizaciones de  salud pueden usar algoritmos  para mejorar la toma de  decisiones tanto a nivel clinico como comercial,  asi podiendo mejorar la calidad de  las experiencias que brinda.
    
Es asi que, en este trabajo se presenta datos para la estimación de los niveles de obesidad en individuos de los países de México, Perú y Colombia, con base en sus hábitos alimentarios y condición física. Los datos contienen 17 atributos y 2111 registros, los registros están etiquetados con la variable de clase NObesidad (Nivel de Obesidad), que permite clasificar los datos utilizando los valores de Peso Insuficiente, Peso Normal, Sobrepeso Nivel I, Sobrepeso Nivel II, Obesidad Tipo I , Obesidad Tipo II y Obesidad Tipo III.
</p>


*Esta herramienta de analisis permite realziar una descripcion general del comportamiento poblacional de las personas para la deteccion de la presencia de diabetes melittus, demo de analitica avanzada.

<h1 align="center"> Analitica de ML para predecir diabetes - </h1>

<p align="center"><img src="https://www.topdoctors.mx/files/Image/large/f0245962b7cd125ebbc6445879251a37.jpg"/></p> 

# Tabla de contenidos:
---

- [Variables_de_estudio](#Variables_de_estudio)
- [Prerequisitos](#Prerequisitos)
- [Analisis_descriptivo](#Analisis_descriptivo)
- [Analisis_bivariado](#Analisis_bivariado)
- [Prueba_de_hipotesis](#Prueba_de_hipotesis)
- [Entrenamiento_modelo_ml_y_evaluacion](#Entrenamiento_modelo_ml_y_evaluacion)
- [Guía_de_usuario](#Guía_de_usuario)

--------------------------------------------------------------------------------------

## Variables_de_estudio

#### hábitos alimenticios:

- `FAVC => Consumo frecuente de alimentos de alto contenido calórico`
- `FCVC => Frecuencia de consumo de verduras`
- `NCP => Número de comidas principales`
- `CAEC => Consumo de alimentos entre horas`
- `CH20 => Consumo de agua diario`
- `CALC => Consumo de alcohol`

#### atributo físico:

- `SCC => Seguimiento del consumo de calorías`
- `FAF => Frecuencia de actividad física`
- `TUE => Tiempo usando dispositivos tecnológicos`
- `MTRANS => Transporte utilizado`

#### otro atributo:

- `GÉNERO`
- `AÑOS`
- `ALTURA`
- `PESO`

-----------------------------------------------------------------------

## Prerequisitos

Para la ejecución de la herramienta de análisis se necesita installar las  siguientes  dependencias:

- ![numpy](https://img.shields.io/badge/numpy%20-1.24.1-green)
- ![scipy](https://img.shields.io/badge/scipy%20-1.10.0-green)
- ![matplotlib](https://img.shields.io/badge/matplotlib%20-3.5.2-green)
- ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.1.2-green)
- ![joblib](https://img.shields.io/badge/joblib%20-1.2.0-green)
- ![pillow](https://img.shields.io/badge/pillow%20-9.4.0-green)
- ![pandas](https://img.shields.io/badge/pandas%20-1.5.2-green)
- ![seaborn](https://img.shields.io/badge/seaborn%20-0.12.2-green)
- ![pingouin](https://img.shields.io/badge/pingouin%20-0.5.2-green)

## Analisis_descriptivo

Se realizan calculos bases ** ( mediana, desviacion estandar, maximo, minimo, media, porcentajes)** para determinar el comportamiento distribucional de las variables de interes. Adicionalmente, se presentan graficas de barras para ver como son las distribuciones de varaibles categoricas y evaluar si hay desbalanceo de clases. Por otro lado,  se hace un análisis de correlaciones para determinar la asociación directa entre cada una da las variables numericas. esto con el objetivo de determinar que tipo de tranformaciones se deben de considerar en el analisis para el procesamiento de datos.

## Analisis_bivariado
Se realiza cruse entre cada una da las variables relevantes análisadas en el item anterior con referencia a la variable de interes a predecir **NObeyesdad**, con el fin de identificar patrones importantes  dentro de la caracterización poblacional  frente a  la presencia de diabetes mellitus y cada uno de sus estadios.

## Prueba_de_hipotesis 

se realizan pruebas de hipotesis para validar cambios significativos en las varaibles influyentes sobre la condicion de diabetes, tales como peso, imc por cada uno de los niveles de obesidad considerando la división  por genero del paciente,  para ello, se realiza el chequeo de normalidad y variabilidad para determinar si se ajsutan pruebas de hipotesis parametricas o no parametricas (mannwhitney)

## Entrenamiento_modelo_ml_y_evaluacion

Se realzia normalizacion ( variables numericas) y categorizacion ( variables categoricas ) para dividir los datos en un 70 % training y 30 % testing .
se implementa como primera faceta los siguientes modelos ML de clasificación

- DecisionTreeClassifier(),
- RandomForestClassifier(),
- GradientBoostingClassifier(),
- SGDClassifier()

Se realiza la evaluación de cada modelo  y se selecciona lso que presentan un accuracy superior al 85%  , una vez el algoritmo detecte que mdoelos cumplen el umbral, se  realiza el  proceso de optimización parametrica  por medio de  GridSearchCV  con 5 k-folds  en valización cruzada para selccionar el mejor modelo ajustado que sera persistido y utilizado para el proceso de **Medicina Predictiva**.  Se realiza evaluación  por medio de la matrix de confusión para observar cada una de las metricas de evaluo  y se aplica ROC para ver la eficiencia del modelo seleccionado.

## Guía_de_usuario

todo el proceso se presenta en un notebook de jupyter,   cada vez que se tenga data nuevo para reentrenar el modelo, es indispensable ejecutar todo el archivo .ipynd  y persistir de nuevo el modelo.

 en caso de que no sea necesario un entrenamiento continuo,  se debe de cargar el modelo pre-entrenado que se encuentra almacenado en **./Modelo/** y ejecutarlo en un ambiente productivo para su utuilización.
