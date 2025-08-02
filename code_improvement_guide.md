# Code Verbesserungs-Leitfaden für docker_exam Notebook

## 1. Allgemeine Prinzipien

### Code-Organisation
- **Funktionen erstellen**: Wiederholende Code-Blöcke in Funktionen auslagern
- **Dokumentation**: Docstrings für alle Funktionen verwenden
- **Kommentare**: Nur sinnvolle Kommentare, die das "Warum" erklären, nicht das "Was"
- **Konsistente Namensgebung**: Aussagekräftige Variablen- und Funktionsnamen

### Testing und Validierung
- **Assertions verwenden**: Statt print() für Tests
- **Logging**: Strukturierte Ausgaben statt einfache print-Statements
- **Error Handling**: Try-catch Blöcke für robuste Funktionen

## 2. Konkrete Verbesserungen

### Statt:
```python
# Lade Daten
df = pd.read_csv('data.csv')
print(df.shape)
print(df.head())
```

### Besser:
```python
def load_and_validate_data(filepath, expected_columns=None):
    """
    Load data from file and perform basic validation.
    
    Parameters:
    -----------
    filepath : str
        Path to the data file
    expected_columns : list, optional
        List of expected column names
    
    Returns:
    --------
    pd.DataFrame
        Loaded and validated dataframe
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        
        if expected_columns:
            missing_cols = set(expected_columns) - set(df.columns)
            if missing_cols:
                print(f"Warning: Missing expected columns: {missing_cols}")
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
```

## 3. Utility Functions Template

### Exploratory Data Analysis
```python
def perform_eda(df, target_column=None):
    """Comprehensive exploratory data analysis with visualizations."""
    print("=== Exploratory Data Analysis ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Data types:\n{df.dtypes}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Basic statistics:\n{df.describe()}")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Exploratory Data Analysis', fontsize=16)
    
    # Missing values heatmap
    sns.heatmap(df.isnull(), cbar=True, ax=axes[0,0])
    axes[0,0].set_title('Missing Values Pattern')
    
    # Correlation matrix
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0,1])
        axes[0,1].set_title('Correlation Matrix')
    
    plt.tight_layout()
    plt.show()
```

### Dimensionality Reduction (mit Sampling!)
```python
def perform_dimensionality_reduction(X, method='pca', n_components=2, sample_size=1000):
    """
    Perform dimensionality reduction with sampling for large datasets.
    WICHTIG: Verwendet Sampling für t-SNE und UMAP um Fehler zu vermeiden.
    """
    # Sample data for expensive methods
    if method in ['tsne', 'umap'] and len(X) > sample_size:
        sample_indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[sample_indices]
        print(f"Sampling {sample_size} points for {method.upper()} due to computational constraints")
    else:
        X_sample = X
        sample_indices = np.arange(len(X))
    
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(X_sample)-1))
    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=42)
    
    X_reduced = reducer.fit_transform(X_sample)
    return X_reduced, sample_indices
```

### Clustering Evaluation
```python
def evaluate_clustering(X, max_clusters=10):
    """Evaluate clustering performance using multiple metrics."""
    inertias = []
    silhouette_scores = []
    
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))
    
    optimal_k = np.argmax(silhouette_scores) + 2
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    k_range = range(2, max_clusters + 1)
    axes[0].plot(k_range, inertias, 'bo-')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Method for Optimal k')
    axes[0].grid(True)
    
    axes[1].plot(k_range, silhouette_scores, 'ro-')
    axes[1].axvline(x=optimal_k, color='r', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Analysis')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'optimal_k': optimal_k,
        'silhouette_scores': silhouette_scores,
        'inertias': inertias
    }
```

## 4. Testing und Validierung

### Statt:
```python
print("Test: Clustering funktioniert")
print(labels)
```

### Besser:
```python
def validate_clustering_results(labels, X):
    """Validate clustering results with proper metrics."""
    n_clusters = len(np.unique(labels))
    silhouette_avg = silhouette_score(X, labels)
    
    assert n_clusters > 1, "Clustering should produce more than 1 cluster"
    assert silhouette_avg > 0, "Silhouette score should be positive"
    
    print(f"✓ Clustering validation passed:")
    print(f"  - Number of clusters: {n_clusters}")
    print(f"  - Silhouette score: {silhouette_avg:.3f}")
    
    return True
```

## 5. Preprocessing Pipeline

```python
def create_preprocessing_pipeline(numeric_features, categorical_features=None):
    """Create a robust preprocessing pipeline."""
    transformers = [
        ('num', RobustScaler(), numeric_features)
    ]
    
    if categorical_features:
        from sklearn.preprocessing import OneHotEncoder
        transformers.append(
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        )
    
    return ColumnTransformer(transformers=transformers, remainder='drop')
```

## 6. Reporting und Dokumentation

```python
def create_analysis_report(df, results):
    """Create a comprehensive analysis report."""
    print("=== ANALYSIS REPORT ===")
    print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*50)
    
    for section, data in results.items():
        print(f"\n{section.upper()}:")
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {data}")
    
    print("\n" + "="*50)
    print("Analysis completed successfully.")
```

## 7. Notebook-Struktur

### Empfohlene Reihenfolge:
1. **Setup und Imports** - Alle Libraries und Konfiguration
2. **Utility Functions** - Alle Hilfsfunktionen
3. **Data Loading** - Daten laden und validieren
4. **Exploratory Data Analysis** - EDA mit Visualisierungen
5. **Data Preprocessing** - Cleaning und Transformation
6. **Analysis** - Hauptanalyse (Clustering, Classification, etc.)
7. **Results und Validation** - Ergebnisse und Tests
8. **Final Report** - Zusammenfassung und Schlussfolgerungen

## 8. Wichtige Hinweise

- **Keine globalen Variablen**: Alles über Funktionsparameter übergeben
- **Konsistente Random States**: Für Reproduzierbarkeit (random_state=42)
- **Sampling für große Datasets**: Besonders bei t-SNE und UMAP
- **Error Handling**: Try-catch für robuste Funktionen
- **Dokumentation**: Docstrings für alle Funktionen
- **Validierung**: Assertions statt print für Tests

## 9. Beispiel für verbesserte Analyse

```python
# Hauptanalyse-Workflow
def run_complete_analysis(data_path):
    """Run complete data analysis workflow."""
    # 1. Load and validate data
    df = load_and_validate_data(data_path)
    if df is None:
        return None
    
    # 2. Exploratory analysis
    perform_eda(df)
    
    # 3. Preprocessing
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    preprocessor = create_preprocessing_pipeline(numeric_features)
    X_processed = preprocessor.fit_transform(df)
    
    # 4. Dimensionality reduction
    X_pca, _ = perform_dimensionality_reduction(X_processed, method='pca')
    X_tsne, _ = perform_dimensionality_reduction(X_processed, method='tsne', sample_size=1000)
    
    # 5. Clustering analysis
    clustering_results = evaluate_clustering(X_processed)
    
    # 6. Final validation and reporting
    optimal_k = clustering_results['optimal_k']
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
    labels = kmeans_final.fit_predict(X_processed)
    
    validate_clustering_results(labels, X_processed)
    
    # 7. Create comprehensive report
    results = {
        'data_quality': {
            'samples': len(df),
            'features': len(df.columns),
            'missing_values': df.isnull().sum().sum()
        },
        'clustering': {
            'optimal_k': optimal_k,
            'silhouette_score': silhouette_score(X_processed, labels)
        }
    }
    
    create_analysis_report(df, results)
    
    return results

# Verwendung:
# results = run_complete_analysis('your_data.csv')
```

Dieser Leitfaden zeigt dir, wie du deinen Code professionell strukturieren kannst. Wenn du dein ursprüngliches Notebook hochlädst, kann ich diese Prinzipien direkt auf deinen Code anwenden.