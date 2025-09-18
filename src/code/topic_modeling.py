import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import warnings
warnings.filterwarnings('ignore')

# Text preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import umap

# BERTopic
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    BERTOPIC_AVAILABLE = True
    print("✓ BERTopic available")
except ImportError:
    BERTOPIC_AVAILABLE = False
    print("❌ BERTopic not available. Install with: pip install bertopic")

# Text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True) 
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

class LegalTopicModeling:
    """Topic modeling for legal case analysis"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        
        # Legal-specific stop words
        self.legal_stopwords = set(stopwords.words('english')).union({
            'case', 'court', 'judge', 'plaintiff', 'defendant', 'v', 'vs',
            'state', 'county', 'city', 'inc', 'corp', 'ltd', 'llc',
            'opinion', 'decision', 'order', 'holding', 'finding',
            'section', 'subsection', 'paragraph', 'clause', 'page',
            'see', 'also', 'however', 'therefore', 'furthermore',
            'moreover', 'nevertheless', 'accordingly', 'thus',
            'one', 'two', 'three', 'first', 'second', 'third',
            'would', 'could', 'should', 'may', 'must', 'shall',
            'said', 'pursuant', 'herein', 'thereof', 'whereas'
        })
        
        # Legal domain patterns to preserve
        self.legal_patterns = [
            r'\b(?:constitutional|criminal|civil|contract|tort|property)\b',
            r'\b(?:due process|equal protection|probable cause|search and seizure)\b',
            r'\b(?:first amendment|second amendment|fourth amendment|fifth amendment)\b',
            r'\b(?:evidence|testimony|discovery|motion|appeal|jurisdiction)\b',
            r'\b(?:liability|damages|injunction|remedy|statute|regulation)\b',
            r'\b(?:precedent|stare decisis|dicta|ratio decidendi)\b'
        ]
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess legal text for topic modeling"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove citations (e.g., "123 F.3d 456", "2023 WL 123456")
        text = re.sub(r'\b\d+\s+[a-z\.]+\s*\d+[a-z]*\b', '', text)
        text = re.sub(r'\b\d{4}\s+[a-z]{2}\s+\d+\b', '', text)
        
        # Remove case names in parentheses
        text = re.sub(r'\([^)]*v[^)]*\)', '', text)
        
        # Remove excessive whitespace and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Tokenize and lemmatize
        tokens = word_tokenize(text)
        
        # Filter tokens
        processed_tokens = []
        for token in tokens:
            # Skip short tokens and numbers
            if len(token) < 3 or token.isdigit():
                continue
            
            # Skip stop words
            if token in self.legal_stopwords:
                continue
            
            # Lemmatize
            lemmatized = self.lemmatizer.lemmatize(token)
            processed_tokens.append(lemmatized)
        
        return ' '.join(processed_tokens)
    
    def prepare_documents(self, df: pd.DataFrame, text_column: str = 'opinion_text') -> List[str]:
        """Prepare documents for topic modeling"""
        print("📝 Preprocessing documents...")
        
        documents = []
        for idx, text in enumerate(df[text_column]):
            processed = self.preprocess_text(text)
            if len(processed.split()) > 10:  # Filter very short documents
                documents.append(processed)
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} documents...")
        
        print(f"✓ Prepared {len(documents)} documents for topic modeling")
        return documents
    
    def fit_lda_model(self, documents: List[str], n_topics: int = 10, 
                      max_features: int = 1000) -> Tuple[LatentDirichletAllocation, CountVectorizer, np.ndarray]:
        """Fit LDA topic model"""
        print(f"🔍 Fitting LDA model with {n_topics} topics...")
        
        # Vectorize documents
        vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=2,
            max_df=0.8,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        doc_term_matrix = vectorizer.fit_transform(documents)
        
        # Fit LDA
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=100,
            learning_method='online',
            learning_offset=50.0
        )
        
        lda_output = lda.fit_transform(doc_term_matrix)
        
        print("✓ LDA model fitted successfully")
        return lda, vectorizer, lda_output
    
    def get_lda_topics(self, lda: LatentDirichletAllocation, 
                       vectorizer: CountVectorizer, n_words: int = 10) -> List[Dict]:
        """Extract topics from LDA model"""
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            top_weights = [topic[i] for i in top_words_idx]
            
            topics.append({
                'topic_id': topic_idx,
                'words': top_words,
                'weights': top_weights,
                'top_words_string': ', '.join(top_words[:5])
            })
        
        return topics
    
    def fit_bertopic_model(self, documents: List[str], min_topic_size: int = 10) -> Optional[BERTopic]:
        """Fit BERTopic model"""
        if not BERTOPIC_AVAILABLE:
            print("❌ BERTopic not available")
            return None
        
        print(f"🤖 Fitting BERTopic model...")
        
        # Initialize sentence transformer
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize BERTopic with UMAP and HDBSCAN
        topic_model = BERTopic(
            embedding_model=sentence_model,
            min_topic_size=min_topic_size,
            calculate_probabilities=True,
            verbose=True
        )
        
        # Fit model
        topics, probs = topic_model.fit_transform(documents)
        
        print(f"✓ BERTopic model fitted with {len(set(topics))} topics")
        return topic_model
    
    def analyze_topic_coherence(self, lda: LatentDirichletAllocation, 
                               doc_term_matrix, n_topics_range: range = range(5, 21)) -> Dict:
        """Analyze topic coherence for different numbers of topics"""
        print("📊 Analyzing topic coherence...")
        
        coherence_scores = []
        perplexity_scores = []
        
        for n_topics in n_topics_range:
            lda_temp = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=50
            )
            lda_temp.fit(doc_term_matrix)
            
            perplexity = lda_temp.perplexity(doc_term_matrix)
            perplexity_scores.append(perplexity)
            
        return {
            'n_topics': list(n_topics_range),
            'perplexity_scores': perplexity_scores
        }
    
    def visualize_lda_topics(self, topics: List[Dict], save_plot: bool = True):
        """Visualize LDA topics"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Topic word clouds (top words bar chart)
        n_topics_to_show = min(4, len(topics))
        for i in range(n_topics_to_show):
            ax = axes[i//2, i%2]
            topic = topics[i]
            
            words = topic['words'][:8]
            weights = topic['weights'][:8]
            
            ax.barh(words, weights)
            ax.set_title(f'Topic {i}: {topic["top_words_string"]}', fontsize=12)
            ax.set_xlabel('Weight')
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('lda_topics.png', dpi=300, bbox_inches='tight')
            print("📊 LDA topics visualization saved as 'lda_topics.png'")
        
        plt.show()
    
    def visualize_topic_distribution(self, lda_output: np.ndarray, topics: List[Dict], 
                                   save_plot: bool = True):
        """Visualize topic distribution across documents"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Average topic weights
        avg_weights = np.mean(lda_output, axis=0)
        topic_labels = [f"T{i}: {topics[i]['top_words_string'][:20]}..." 
                       for i in range(len(topics))]
        
        axes[0, 0].bar(range(len(avg_weights)), avg_weights)
        axes[0, 0].set_title('Average Topic Weights Across Documents')
        axes[0, 0].set_xlabel('Topic')
        axes[0, 0].set_ylabel('Average Weight')
        axes[0, 0].set_xticks(range(len(topic_labels)))
        axes[0, 0].set_xticklabels([f"T{i}" for i in range(len(topics))], rotation=45)
        
        # 2. Topic distribution heatmap
        if lda_output.shape[0] > 20:
            sample_docs = lda_output[:20]  # Show first 20 documents
        else:
            sample_docs = lda_output
        
        im = axes[0, 1].imshow(sample_docs.T, cmap='Blues', aspect='auto')
        axes[0, 1].set_title('Topic Distribution Across Sample Documents')
        axes[0, 1].set_xlabel('Document')
        axes[0, 1].set_ylabel('Topic')
        plt.colorbar(im, ax=axes[0, 1])
        
        # 3. Document-topic assignment (dominant topic)
        dominant_topics = np.argmax(lda_output, axis=1)
        topic_counts = Counter(dominant_topics)
        
        topics_list = list(topic_counts.keys())
        counts_list = list(topic_counts.values())
        
        axes[1, 0].bar(topics_list, counts_list)
        axes[1, 0].set_title('Document Count by Dominant Topic')
        axes[1, 0].set_xlabel('Topic')
        axes[1, 0].set_ylabel('Number of Documents')
        
        # 4. Topic coherence distribution
        topic_coherences = []
        for i in range(lda_output.shape[1]):
            topic_docs = lda_output[:, i]
            coherence = np.std(topic_docs)  # Simple coherence measure
            topic_coherences.append(coherence)
        
        axes[1, 1].bar(range(len(topic_coherences)), topic_coherences)
        axes[1, 1].set_title('Topic Coherence (Standard Deviation)')
        axes[1, 1].set_xlabel('Topic')
        axes[1, 1].set_ylabel('Coherence Score')
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('topic_distribution.png', dpi=300, bbox_inches='tight')
            print("📊 Topic distribution saved as 'topic_distribution.png'")
        
        plt.show()
    
    def visualize_bertopic_results(self, topic_model: BERTopic, documents: List[str], 
                                 save_plot: bool = True):
        """Visualize BERTopic results"""
        if not topic_model:
            return
        
        try:
            # Topic visualization
            fig1 = topic_model.visualize_topics()
            if save_plot:
                fig1.write_html("bertopic_topics.html")
                print("📊 BERTopic topics saved as 'bertopic_topics.html'")
            fig1.show()
            
            # Topic hierarchy
            hierarchical_topics = topic_model.hierarchical_topics(documents)
            fig2 = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
            if save_plot:
                fig2.write_html("bertopic_hierarchy.html")
                print("📊 BERTopic hierarchy saved as 'bertopic_hierarchy.html'")
            fig2.show()
            
            # Topic heatmap
            fig3 = topic_model.visualize_heatmap()
            if save_plot:
                fig3.write_html("bertopic_heatmap.html")
                print("📊 BERTopic heatmap saved as 'bertopic_heatmap.html'")
            fig3.show()
            
        except Exception as e:
            print(f"Error creating BERTopic visualizations: {e}")
    
    def cluster_similar_cases(self, lda_output: np.ndarray, df: pd.DataFrame, 
                            n_clusters: int = 5) -> Dict:
        """Cluster cases based on topic distributions"""
        print(f"🔍 Clustering cases into {n_clusters} clusters...")
        
        # K-means clustering on topic distributions
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(lda_output)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(lda_output, cluster_labels)
        
        # Analyze clusters
        cluster_analysis = {
            'n_clusters': n_clusters,
            'silhouette_score': silhouette_avg,
            'cluster_sizes': Counter(cluster_labels),
            'clusters': {}
        }
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_docs = lda_output[cluster_mask]
            
            # Get cluster characteristics
            avg_topic_dist = np.mean(cluster_docs, axis=0)
            dominant_topic = np.argmax(avg_topic_dist)
            
            # Get sample cases from this cluster
            cluster_indices = np.where(cluster_mask)[0]
            sample_cases = []
            
            for idx in cluster_indices[:3]:  # Get first 3 cases as samples
                if idx < len(df):
                    case_info = {
                        'case_name': df.iloc[idx].get('case_name_short', 'Unknown'),
                        'court_type': df.iloc[idx].get('court_type', 'Unknown'),
                        'year': df.iloc[idx].get('decision_date', 'Unknown')
                    }
                    sample_cases.append(case_info)
            
            cluster_analysis['clusters'][cluster_id] = {
                'size': int(np.sum(cluster_mask)),
                'dominant_topic': int(dominant_topic),
                'avg_topic_distribution': avg_topic_dist.tolist(),
                'sample_cases': sample_cases
            }
        
        print(f"✓ Clustering complete (Silhouette score: {silhouette_avg:.3f})")
        return cluster_analysis
    
    def save_results(self, lda_topics: List[Dict], cluster_analysis: Dict,
                    bertopic_model: Optional[BERTopic] = None, 
                    output_dir: str = "data/processed/"):
        """Save topic modeling results"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        def convert_numpy_types(obj):
            """Convert numpy types to Python native types for JSON serialization"""
            if isinstance(obj, dict):
                return {str(k): convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                # Handle NaN values
                if np.isnan(obj):
                    return None
                return float(obj)
            elif isinstance(obj, np.ndarray):
                # Handle NaN values in arrays
                clean_array = obj.tolist()
                return clean_nan_values(clean_array)
            elif obj is np.nan or (isinstance(obj, float) and np.isnan(obj)):
                return None
            else:
                return obj
        
        def clean_nan_values(obj):
            """Recursively clean NaN values from nested structures"""
            if isinstance(obj, list):
                return [clean_nan_values(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: clean_nan_values(v) for k, v in obj.items()}
            elif isinstance(obj, float) and np.isnan(obj):
                return None
            else:
                return obj
        
        # Convert numpy types in the data
        clean_lda_results = convert_numpy_types({
            'topics': lda_topics,
            'cluster_analysis': cluster_analysis
        })
        
        lda_file = os.path.join(output_dir, "lda_topic_modeling.json")
        with open(lda_file, 'w') as f:
            json.dump(clean_lda_results, f, indent=2)
        print(f"💾 LDA results saved: {lda_file}")
        
        # Save BERTopic results
        if bertopic_model:
            bertopic_file = os.path.join(output_dir, "bertopic_model")
            bertopic_model.save(bertopic_file)
            print(f"💾 BERTopic model saved: {bertopic_file}")
            
            # Save topic info
            topics_info = bertopic_model.get_topic_info()
            topics_file = os.path.join(output_dir, "bertopic_topics.csv")
            topics_info.to_csv(topics_file, index=False)
            print(f"💾 BERTopic topics saved: {topics_file}")

def load_data(file_path: str) -> pd.DataFrame:
    """Load the legal cases dataset"""
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Dataset loaded: {df.shape[0]} cases, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return pd.DataFrame()

def main():
    """Main execution function"""
    print("🚀 Legal Topic Modeling & Case Clustering")
    print("="*60)
    
    # Initialize topic modeling
    topic_modeler = LegalTopicModeling()
    
    # Load data
    data_file = "Text-mining-HTW/data/processed/cleaned/parse_legal_cases.csv"
    df = load_data(data_file)
    
    if df.empty:
        print("❌ No data loaded. Please check the file path.")
        return
    
    # Limit to subset for processing efficiency
    max_cases = min(500, len(df))
    df_subset = df.head(max_cases).copy()
    print(f"Processing {len(df_subset)} cases for topic modeling")
    
    # Prepare documents
    documents = topic_modeler.prepare_documents(df_subset, 'opinion_text')
    
    if len(documents) < 10:
        print("❌ Insufficient documents for topic modeling")
        return
    
    print(f"\n📊 Starting topic modeling with {len(documents)} documents...")
    
    # LDA Topic Modeling
    print("\n" + "="*40)
    print("LDA TOPIC MODELING")
    print("="*40)
    
    # Find optimal number of topics
    lda, vectorizer, lda_output = topic_modeler.fit_lda_model(
        documents, n_topics=10, max_features=1000
    )
    
    # Extract topics
    lda_topics = topic_modeler.get_lda_topics(lda, vectorizer, n_words=10)
    
    # Print LDA topics
    print("\n🔍 LDA TOPICS:")
    for topic in lda_topics:
        print(f"Topic {topic['topic_id']}: {topic['top_words_string']}")
    
    # Visualize LDA results
    print(f"\n📊 Creating LDA visualizations...")
    topic_modeler.visualize_lda_topics(lda_topics)
    topic_modeler.visualize_topic_distribution(lda_output, lda_topics)
    
    # Cluster similar cases
    print(f"\n🔍 Clustering similar legal cases...")
    cluster_analysis = topic_modeler.cluster_similar_cases(lda_output, df_subset, n_clusters=5)
    
    print(f"\n📋 CLUSTER ANALYSIS:")
    print(f"Silhouette Score: {cluster_analysis['silhouette_score']:.3f}")
    
    for cluster_id, cluster_info in cluster_analysis['clusters'].items():
        print(f"\nCluster {cluster_id} (Size: {cluster_info['size']}):")
        print(f"  Dominant Topic: {cluster_info['dominant_topic']}")
        if cluster_info['sample_cases']:
            print("  Sample Cases:")
            for case in cluster_info['sample_cases']:
                print(f"    - {case['case_name']} ({case['court_type']})")
    
    # BERTopic Modeling
    if BERTOPIC_AVAILABLE:
        print("\n" + "="*40)
        print("BERTOPIC MODELING")
        print("="*40)
        
        bertopic_model = topic_modeler.fit_bertopic_model(documents, min_topic_size=5)
        
        if bertopic_model:
            # Print BERTopic topics
            topic_info = bertopic_model.get_topic_info()
            print(f"\n🔍 BERTOPIC TOPICS ({len(topic_info)} topics found):")
            
            for _, row in topic_info.head(10).iterrows():
                topic_id = row['Topic']
                if topic_id != -1:  # Skip outlier topic
                    words = bertopic_model.get_topic(topic_id)
                    top_words = [word for word, _ in words[:5]]
                    print(f"Topic {topic_id}: {', '.join(top_words)}")
            
            # Create BERTopic visualizations
            print(f"\n📊 Creating BERTopic visualizations...")
            topic_modeler.visualize_bertopic_results(bertopic_model, documents)
    else:
        bertopic_model = None
        print("\n❌ BERTopic not available. Install with: pip install bertopic")
    
    # Save results
    print(f"\n💾 Saving results...")
    topic_modeler.save_results(lda_topics, cluster_analysis, bertopic_model)
    
    print(f"\n✅ Topic modeling complete!")
    print(f"📊 Found {len(lda_topics)} LDA topics")
    if bertopic_model:
        n_bertopics = len(bertopic_model.get_topic_info()) - 1  # Exclude outlier topic
        print(f"🤖 Found {n_bertopics} BERTopic topics")
    print(f"🔍 Identified {cluster_analysis['n_clusters']} case clusters")

if __name__ == "__main__":
    main()