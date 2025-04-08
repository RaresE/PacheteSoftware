import math
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
from pandas.core.dtypes.common import is_numeric_dtype
import io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

date = pd.read_csv("winequality-red.csv")
df = date.iloc[:500,:] #modificam daca ne dormim mai multe date din fisier
date = date.iloc[:50,:]
date = pd.DataFrame(date)
df = pd.DataFrame(df)

section = st.sidebar.selectbox("Selectează pagina:", ["Afișare Date", "Descriere Date"])


st.markdown(
    """
    <style>
    .custom-title {
        color: #80091B !important;
        font-size: 40px;
        text-align: center;
    }
    
    h2.custom-subtitle {
        color: #FFFFFF !important;
        font-size: 17px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="custom-title">Proiect Pachete Software</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="custom-subtitle">Realizat de Enache Rareș și Ionescu Tudor</h2>', unsafe_allow_html=True)

if section == "Afișare Date":
    st.header("Setul de date")
    st.dataframe(date, width=1000)

    missing_values = df.isnull().sum()
    if missing_values.sum() == 0:
        st.success("Nu există valori lipsă în dataset! 🎉")
    else:
        st.warning("Există valori lipsă în următoarele coloane:")
        st.write(missing_values[missing_values > 0])

    def nan_replace_t(t):
        assert isinstance(t, pd.DataFrame)
        for v in t.columns:
            if any(t[v].isna()):
                if is_numeric_dtype(t[v]):
                    t.fillna({v: t[v].mean()}, inplace=True)
                else:
                    t.fillna({v: t[v].mode()[0]}, inplace=True)

    st.subheader("Primele 5 rânduri din dataset")
    st.dataframe(df.head())

    st.subheader("Informații generale despre dataset")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.subheader("Informații folosind descriptive statistics")
    st.dataframe(df.describe())

    st.subheader("Caracteristicile vinurilor cu cea mai slabă calitate")
    st.dataframe(df.loc[df['quality'].isin([1,4])])

    st.subheader("Selectare ultima coloană (quality_category)")
    def categorize_quality(quality):
        if quality <= 5:
            return 'low quality'
        elif quality <= 7:
            return 'medium quality'
        else:
            return 'high quality'

    df['quality_category'] = df['quality'].apply(categorize_quality)
    st.dataframe(df.iloc[:, -1])

    st.subheader("Piechart pentru variabila 'quality_category'")
    fig, ax = plt.subplots()
    quality_counts = df['quality_category'].value_counts()
    ax.pie(quality_counts, labels=quality_counts.index, autopct='%1.1f%%', startangle=90)
    ax.set_title("Distribuția categoriilor de calitate")
    st.pyplot(fig)

    st.subheader("Histogramă pentru variabila 'alcohol'")
    fig, ax = plt.subplots()
    ax.hist(df['alcohol'], bins=20, color='#80091B', edgecolor='black')
    ax.set_title("Distribuția alcoolului în vinuri")
    ax.set_xlabel('alcohol (%)')
    ax.set_ylabel('Frecvență')
    st.pyplot(fig)

    st.subheader("Generarea histogramelor pentru variabilele numerice")
    n_cols = 3
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    #print(numerical_cols)
    n_rows = math.ceil(
        len(numerical_cols) / n_cols)  # Calculăm numărul de rânduri necesare în funcție de numărul total de coloane numerice
    plt.figure(figsize=(6 * n_cols, 4 * n_rows))  # Setăm dimensiunea totală a figurii (lățime și înălțime)

    # Iterăm prin fiecare coloană numerică și generăm histogramă
    for i, col in enumerate(numerical_cols):
        plt.subplot(n_rows, n_cols,
                    i + 1)  # Creăm un subplot în grila de n_rows x n_cols; i+1 pentru indexarea subgraficelor începând de la 1
        plt.hist(df[col].dropna(), bins=30, edgecolor='black',
                 color='skyblue')  # Construim histograma pentru coloana curentă, eliminând valorile lipsă
        plt.title(f'Distribuția: {col}')  # Setăm titlul graficului cu numele coloanei
        plt.xlabel(col)  # Etichetă pentru axa x, indicând numele variabilei
        plt.ylabel('Frecvență')  # Etichetă pentru axa y, indicând frecvența valorilor
    plt.tight_layout()  # Ajustăm automat spațiile dintre subgrafice pentru a evita suprapunerea
    st.pyplot(plt.gcf())

    st.subheader("Repartizarea calităților vinurilor pe categorii")
    # Iterăm prin fiecare coloană de tip obiect (categorică)
    plt.figure(figsize=(8, 4))  # Creăm o figură nouă pentru fiecare coloană categorică
    unique_count = df["quality_category"].nunique()  # Calculăm numărul de valori unice din coloana curentă
    # Dacă numărul de categorii este mic, construim direct un countplot
    sns.countplot(x="quality_category", data=df, palette='viridis')
    plt.title(f'Distribuția: {"quality_category"}')  # Setăm titlul graficului
    plt.xlabel("quality_category")  # Etichetă pentru axa x
    plt.ylabel('Frecvență')  # Etichetă pentru axa y
    plt.xticks(rotation=45)  # Rotim etichetele de pe axa x
    plt.tight_layout()  # Ajustăm automat spațiile în figură
    st.pyplot(plt.gcf())  # Afișăm figura pentru coloana categorică

    st.subheader("Distribuția conținutului de alcool (grupat în intervale)")
    if 'alcohol' in df.columns:  # Verificăm dacă există coloana 'alcohol'
        # Creăm categorii de alcool folosind pd.cut
        df['alcool_interval'] = pd.cut(df['alcohol'], bins=[0, 8, 10, 12, 14, 20],
                                       labels=["<8%", "8-10%", "10-12%", "12-14%", ">14%"])

        plt.figure(figsize=(8, 4))  # Dimensiuni figură
        sns.countplot(x='alcool_interval', data=df,
                      palette='coolwarm')  # Construim un countplot pe baza intervalelor de alcool

        plt.title("Distribuția conținutului de alcool")  # Titlu grafic
        plt.xlabel("Interval Alcool (%)")  # Etichetă axa x
        plt.ylabel("Număr de mostre")  # Etichetă axa y
        plt.xticks(rotation=45)  # Rotim etichetele axei x
        plt.tight_layout()  # Ajustare automată a spațiului
        st.pyplot(plt.gcf())  # Afișăm graficul

    st.subheader("Pairplot pentru variabilele numerice")
    def plot_pairplot_numeric(df, numerical_cols):
        """
        Creează un pairplot pentru variabilele numerice.
        - diag_kind='kde' -> pe diagonală se afișează grafic de densitate
        - corner=True -> afișează doar jumătate din matrice (opțional)
        """
        fig = sns.pairplot(df[numerical_cols], diag_kind='kde')
        fig.fig.suptitle("Pairplot pentru variabilele numerice", y=1.02)
        st.pyplot(fig)
    plot_pairplot_numeric(df, numerical_cols)

    st.subheader("🍷 Vizualizare Boxplot: Calitate vs. Caracteristici vin")
    # Selectează coloana numerică pentru boxplot
    selected_col = st.selectbox("Alege o caracteristică numerică pentru boxplot:", numerical_cols)
    # Funcția de generare boxplot
    def plot_boxplot(df, num_col):
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=df, x='quality_category', y=num_col, palette='coolwarm', ax=ax)
        ax.set_title(f"Distribuția valorii '{num_col}' în funcție de calitatea vinului")
        ax.set_xlabel("Calitatea vinului")
        ax.set_ylabel(num_col)
        st.pyplot(fig)
    # Afișează boxplot-ul
    plot_boxplot(df, selected_col)

    st.subheader("🔍 Detectare Outlieri cu Metoda IQR")

    def find_outliers_iqr(df, col):
        """
        Identifică outlierii într-o coloană numerică folosind Metoda IQR.
        Returnează un DataFrame cu outlierii și limitele IQR.
        """
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_df = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

        return outliers_df, Q1, Q3, IQR, lower_bound, upper_bound

    selected_outlier_col = st.selectbox("Selectează o coloană numerică pentru analiză:", numerical_cols)
    if selected_outlier_col:
        outliers, Q1, Q3, IQR, lb, ub = find_outliers_iqr(df, selected_outlier_col)

        st.markdown(f"""
        *Q1 (25%):* {Q1:.2f}  
        *Q3 (75%):* {Q3:.2f}  
        *IQR (Q3 - Q1):* {IQR:.2f}  
        *Limita inferioară:* {lb:.2f}  
        *Limita superioară:* {ub:.2f}  
        """)

        st.markdown(f"*Număr outlieri detectați:* {outliers.shape[0]}")
        st.dataframe(outliers)

        # Vizualizare grafică
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df[selected_outlier_col], bins=30, kde=True, color='skyblue', ax=ax)
        ax.axvline(lb, color='red', linestyle='--', label='Limita inferioară')
        ax.axvline(ub, color='red', linestyle='--', label='Limita superioară')
        ax.set_title(f"Distribuția valorilor pentru {selected_outlier_col}")
        ax.set_xlabel(selected_outlier_col)
        ax.set_ylabel("Frecvență")
        ax.legend()
        st.pyplot(fig)

    st.subheader("🔢 Transformare Logaritmică (log1p)")
    log_cols = st.multiselect("Alege coloanele pentru transformare logaritmică:", numerical_cols)
    if log_cols:
        for col in log_cols:
            df[f'log_{col}'] = np.log1p(df[col])

        st.success(f"Am creat coloanele: {['log_' + col for col in log_cols]}")

        st.subheader("📊 Histogramă comparativă: Original vs Log")

        for col in log_cols:
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))

            # Original
            sns.histplot(df[col], bins=30, color='salmon', edgecolor='black', ax=ax[0])
            ax[0].set_title(f"Original: {col}")
            ax[0].set_xlabel(col)

            # Log
            sns.histplot(df[f'log_{col}'], bins=30, color='seagreen', edgecolor='black', ax=ax[1])
            ax[1].set_title(f"Logaritmat: log_{col}")
            ax[1].set_xlabel(f'log_{col}')

            st.pyplot(fig)
        st.subheader("🧾 Primele 5 rânduri din coloanele logaritmate")
        st.dataframe(df[[f'log_{col}' for col in log_cols]].head())

    st.subheader("🔗 Analiza Corelațiilor între Variabile")
    # Extragem coloanele numerice (dacă nu ai făcut-o deja)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Calculăm matricea de corelație
    corr_matrix = df[numerical_cols].corr()
    # Afișăm matricea sub formă de heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Matricea de Corelație pentru variabilele numerice")
    st.pyplot(fig)
    # Poți afișa și DataFrame-ul brut dacă vrei:
    with st.expander("🔎 Vezi matricea de corelație în format tabelar"):
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None))


    st.subheader("🔤 Mapare Label Encoding:")
    # Initializăm encoderul
    le = LabelEncoder()
    # Aplicăm encoding pe coloana 'quality_category'
    df['quality_category_encoded'] = le.fit_transform(df['quality_category'])
    # Afișăm cum s-au mapat valorile
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    st.write(label_mapping)
    # Afișăm primele rânduri din noua coloană
    st.subheader("Coloana 'quality_category_encoded'")
    st.dataframe(df[['quality_category', 'quality_category_encoded']].head())

    st.subheader("⚙ Standardizare și Normalizare a Variabilelor Numerice")

    # Selectăm doar coloanele numerice (fără coloana target dacă ai deja encoding)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_scale = [col for col in numerical_cols if col != 'quality']  # Excludem 'quality' brută

    # Copiem un DataFrame pentru procesare
    #Media 0 si deviatia standard 1
    df_scaled = df.copy()
    # ===== Standardizare =====
    standard_scaler = StandardScaler()
    df_scaled[[f"{col}_standardized" for col in cols_to_scale]] = standard_scaler.fit_transform(df[cols_to_scale])
    # ===== Normalizare =====
    minmax_scaler = MinMaxScaler()
    df_scaled[[f"{col}_normalized" for col in cols_to_scale]] = minmax_scaler.fit_transform(df[cols_to_scale])
    # ===== Confirmăm valorile =====
    st.write("📊 Media și deviația standard după standardizare:")
    for col in cols_to_scale:
        st.write(
            f"- {col}_standardized → μ = {df_scaled[f'{col}_standardized'].mean():.2f}, σ = {df_scaled[f'{col}_standardized'].std():.2f}")
    st.write("📊 Min și max după normalizare:")
    for col in cols_to_scale:
        min_val = df_scaled[f'{col}_normalized'].min()
        max_val = df_scaled[f'{col}_normalized'].max()
        st.write(f"- {col}_normalized → min = {min_val:.2f}, max = {max_val:.2f}")
    # ===== Conversie coloane boolean (dacă există) =====
    bool_cols = df_scaled.select_dtypes(include=['bool']).columns
    df_scaled[bool_cols] = df_scaled[bool_cols].astype(int)
    # ===== Afișăm structura finală =====
    st.subheader("🔍 Primele rânduri din DataFrame-ul procesat")
    st.dataframe(df_scaled.head())

    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix

    st.subheader("Model de Clasificare: Predicția Calității Vinului (Bun / Mediu / Prost)")
    # Pregătim etichetele
    le = LabelEncoder()
    df['quality_encoded'] = le.fit_transform(df['quality_category'])
    # Alegem caracteristicile (toate numerice, fără 'quality' și 'quality_encoded')
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in feature_cols if col not in ['quality', 'quality_encoded']]
    X = df[feature_cols]
    y = df['quality_encoded']
    # Împărțim datele
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Model Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    # Predicții
    y_pred = model.predict(X_test)
    # Evaluare
    st.markdown("### 📊 Clasificare - Calitate vin")
    st.text("Matricea de confuzie:")
    st.text(confusion_matrix(y_test, y_pred))
    st.text("Raport de clasificare:")
    st.text(classification_report(y_test, y_pred, target_names=le.classes_))

    st.subheader("🤖 Clasificare: Predicția Calității Vinului")
    # Convertim 'quality_category' în numeric
    le = LabelEncoder()
    df['quality_encoded'] = le.fit_transform(df['quality_category'])
    # Selectăm doar coloane numerice pentru antrenare
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in feature_cols if col not in ['quality', 'quality_encoded']]
    X = df[feature_cols]
    y = df['quality_encoded']
    # Împărțim datele
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Alegem modelul
    model_choice = st.selectbox("Alege modelul de clasificare:", ["Random Forest", "XGBoost"])
    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_choice == "XGBoost":
        model = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    # Antrenare model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Evaluare
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    st.markdown(f"### 🎯 Acuratețe model: {accuracy:.4f}")
    st.markdown("### 📋 Raport de clasificare")
    st.dataframe(pd.DataFrame(report).transpose())
    # Matrice de confuzie
    cf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel("Predicție")
    ax.set_ylabel("Valori reale")
    ax.set_title(f"{model_choice} - Matrice de Confuzie")
    st.pyplot(fig)

elif section == "Descriere Date":
    st.header("Descrierea setului de date")
    st.write("Acest set de date provine dintr-un studiu realizat de P. Cortez et al., 2009 și este disponibil pe UCI Machine Learning Repository și Kaggle. "
             "Setul de date conține informații despre caracteristicile fizico-chimice ale vinului roșu portughez **Vinho Verde**, precum și scorurile de calitate atribuite acestuia pe baza analizei senzoriale. "
             "Scopul principal al acestui set de date este de a explora relația dintre proprietățile chimice ale vinului și percepția asupra calității sale. "
             "Aceste date pot fi utilizate atât pentru regresie (predicția scorului de calitate), cât și pentru clasificare (împărțirea vinurilor în bune și slabe pe baza unui prag de calitate).")

    st.subheader("Variabile disponibile")
    st.write("**Caracteristici fizico-chimice (input):**")
    st.write("1. **fixed acidity** – Aciditate fixă (ex. acid tartric, nu se evaporă ușor).")
    st.write("2. **volatile acidity** – Aciditate volatilă (ex. acid acetic, care poate da un miros de oțet).")
    st.write("3. **citric acid** – Acid citric (contribuie la prospețimea vinului).")
    st.write("4. **residual sugar** – Zahăr rezidual (nivel mai mare → vin mai dulce).")
    st.write("5. **chlorides** – Conținutul de sare al vinului.")
    st.write("6. **free sulfur dioxide** – Dioxid de sulf liber (protejează vinul împotriva oxidării).")
    st.write("7. **total sulfur dioxide** – Dioxid de sulf total (SO₂ liber + legat).")
    st.write("8. **density** – Densitatea vinului (corelată cu alcoolul și zahărul).")
    st.write("9. **pH** – Aciditatea generală a vinului.")
    st.write("10. **sulphates** – Sulfați (pot afecta stabilitatea și aroma vinului).")
    st.write("11. **alcohol** – Conținutul de alcool (%).")

    st.write("**Variabilă țintă (output):**")
    st.write("12. **quality** – Scor de calitate al vinului, pe o scară de la 0 la 10 (evaluare senzorială).")
