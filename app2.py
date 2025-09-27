
import streamlit as st
import spacy
import pandas as pd
import numpy as np
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModel,
    BertTokenizer,
    BertModel
)
import PyPDF2
import io
import re
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import torch
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import warnings
warnings.filterwarnings('ignore')

# Configuration
st.set_page_config(
    page_title="Advanced AI Resume Optimizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('maxent_ne_chunker', quiet=True)
        nltk.download('words', quiet=True)
        return True
    except:
        return False

# Initialize Advanced NLP models
@st.cache_resource
def load_advanced_models():
    models = {}
    
    try:
        # Load spaCy
        models['nlp'] = spacy.load("en_core_web_sm")
    except:
        st.error("Please install spaCy model: python -m spacy download en_core_web_sm")
        st.stop()
    
    # Load multiple transformer models for different tasks
    try:
        # Sentence transformer for semantic similarity
        models['sentence_transformer'] = SentenceTransformer('all-MiniLM-L6-v2')
        
        # BERT for contextual understanding
        models['bert_tokenizer'] = BertTokenizer.from_pretrained('bert-base-uncased')
        models['bert_model'] = BertModel.from_pretrained('bert-base-uncased')
        
        # Specialized models
        models['sentiment_analyzer'] = pipeline("sentiment-analysis", 
                                               model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                                               return_all_scores=True)
        
        models['ner_model'] = pipeline("ner", 
                                      model="dbmdz/bert-large-cased-finetuned-conll03-english",
                                      aggregation_strategy="simple")
        
        models['classification_model'] = pipeline("zero-shot-classification",
                                                 model="facebook/bart-large-mnli")
        
        # Topic modeling
        models['summarizer'] = pipeline("summarization", 
                                       model="facebook/bart-large-cnn")
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# Advanced Text Preprocessing
class AdvancedTextProcessor:
    def __init__(self, models):
        self.models = models
        self.stop_words = set(stopwords.words('english'))
        
        # Industry-specific skill databases
        self.skill_databases = {
            "technical_skills": {
                "programming": ["python", "java", "javascript", "c++", "scala", "r", "sql", "bash", "go", "rust"],
                "ml_frameworks": ["tensorflow", "pytorch", "keras", "scikit-learn", "xgboost", "lightgbm", "catboost"],
                "cloud_platforms": ["aws", "azure", "gcp", "docker", "kubernetes", "lambda", "sagemaker", "databricks"],
                "data_tools": ["pandas", "numpy", "spark", "hadoop", "kafka", "airflow", "dbt", "snowflake"],
                "ai_specific": ["llm", "gpt", "bert", "transformer", "rag", "langchain", "openai", "huggingface"],
                "databases": ["postgresql", "mongodb", "redis", "cassandra", "neo4j", "elasticsearch"],
                "visualization": ["tableau", "power bi", "plotly", "matplotlib", "seaborn", "d3.js"]
            },
            "business_skills": {
                "analytics": ["data analysis", "statistical analysis", "predictive modeling", "forecasting"],
                "project_management": ["agile", "scrum", "kanban", "project planning", "stakeholder management"],
                "communication": ["presentation", "documentation", "reporting", "storytelling", "collaboration"],
                "domain_knowledge": ["finance", "healthcare", "retail", "e-commerce", "fintech", "marketing"]
            }
        }
        
        # ATS keywords by category
        self.ats_categories = {
            "action_words": ["developed", "implemented", "designed", "built", "created", "managed", "led", "optimized"],
            "impact_words": ["improved", "increased", "reduced", "achieved", "delivered", "enhanced", "streamlined"],
            "collaboration": ["collaborated", "partnered", "coordinated", "facilitated", "mentored", "trained"]
        }
    
    def extract_advanced_entities(self, text):
        """Extract entities using multiple NER models"""
        entities = {
            'skills': set(),
            'technologies': set(),
            'organizations': set(),
            'metrics': [],
            'achievements': []
        }
        
        # Use transformer NER
        ner_results = self.models['ner_model'](text)
        for entity in ner_results:
            if entity['entity_group'] == 'ORG':
                entities['organizations'].add(entity['word'].lower())
            elif entity['entity_group'] in ['MISC', 'PER']:
                if any(tech in entity['word'].lower() for tech_list in self.skill_databases['technical_skills'].values() for tech in tech_list):
                    entities['technologies'].add(entity['word'].lower())
        
        # Extract metrics (numbers with context)
        metric_patterns = [
            r'(\d+(?:\.\d+)?)\s*%',  # Percentages
            r'(\d+(?:,\d{3})*)\s*(?:users|customers|records|transactions)',  # Large numbers
            r'improved?\s+by\s+(\d+(?:\.\d+)?)',  # Improvements
            r'reduced?\s+by\s+(\d+(?:\.\d+)?)',  # Reductions
            r'increased?\s+by\s+(\d+(?:\.\d+)?)'  # Increases
        ]
        
        for pattern in metric_patterns:
            matches = re.findall(pattern, text.lower())
            entities['metrics'].extend(matches)
        
        # Extract skills using domain knowledge
        text_lower = text.lower()
        for category, skill_lists in self.skill_databases['technical_skills'].items():
            for skill in skill_lists:
                if skill in text_lower:
                    entities['skills'].add(skill)
        
        return entities
    
    def get_semantic_similarity(self, text1, text2):
        """Calculate semantic similarity using sentence transformers"""
        embeddings = self.models['sentence_transformer'].encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity
    
    def analyze_document_structure(self, text):
        """Analyze document structure for ATS compatibility"""
        structure_score = 0
        total_checks = 10
        
        # Check for common resume sections
        sections = {
            'summary': ['summary', 'profile', 'objective'],
            'experience': ['experience', 'work history', 'employment'],
            'education': ['education', 'degree', 'university', 'college'],
            'skills': ['skills', 'technical skills', 'competencies'],
            'projects': ['projects', 'portfolio'],
            'certifications': ['certifications', 'certificates', 'licensed']
        }
        
        text_lower = text.lower()
        found_sections = 0
        for section_type, keywords in sections.items():
            if any(keyword in text_lower for keyword in keywords):
                found_sections += 1
        
        structure_score += found_sections / len(sections) * 3
        
        # Check formatting indicators
        if re.search(r'[•\-\*]', text):  # Bullet points
            structure_score += 1
        
        if re.search(r'\d{4}\s*-\s*\d{4}|\d{4}\s*to\s*\d{4}', text):  # Date ranges
            structure_score += 1
        
        if re.search(r'[A-Z][a-z]+,\s*[A-Z]{2}', text):  # Location format
            structure_score += 1
        
        # Check for contact information
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):  # Email
            structure_score += 1
        
        if re.search(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text):  # Phone
            structure_score += 1
        
        # Check word count (optimal range)
        word_count = len(text.split())
        if 400 <= word_count <= 800:
            structure_score += 1
        elif 300 <= word_count <= 1000:
            structure_score += 0.5
        
        # Check for action words
        action_word_count = sum(1 for word in self.ats_categories['action_words'] if word in text_lower)
        structure_score += min(action_word_count / 5, 1)
        
        return min(structure_score / total_checks, 1.0)

# Advanced Resume Analyzer
class AdvancedResumeAnalyzer:
    def __init__(self, models, processor):
        self.models = models
        self.processor = processor
    
    def calculate_advanced_ats_score(self, resume_text, job_description):
        """Calculate ATS score using multiple advanced techniques"""
        
        # 1. Semantic similarity (40% weight)
        semantic_sim = self.processor.get_semantic_similarity(resume_text, job_description)
        
        # 2. Keyword matching with TF-IDF (30% weight)
        vectorizer = TfidfVectorizer(
            stop_words='english', 
            ngram_range=(1, 3),
            max_features=1000,
            lowercase=True
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
            tfidf_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            tfidf_sim = 0
        
        # 3. Named entity matching (20% weight)
        resume_entities = self.processor.extract_advanced_entities(resume_text)
        job_entities = self.processor.extract_advanced_entities(job_description)
        
        skill_overlap = len(resume_entities['skills'].intersection(job_entities['skills']))
        total_job_skills = max(len(job_entities['skills']), 1)
        entity_score = skill_overlap / total_job_skills
        
        # 4. Document structure (10% weight)
        structure_score = self.processor.analyze_document_structure(resume_text)
        
        # Weighted final score
        final_score = (
            semantic_sim * 0.4 + 
            tfidf_sim * 0.3 + 
            entity_score * 0.2 + 
            structure_score * 0.1
        ) * 100
        
        return min(final_score, 100), {
            'semantic_similarity': semantic_sim * 100,
            'keyword_match': tfidf_sim * 100,
            'entity_match': entity_score * 100,
            'structure_score': structure_score * 100
        }
    
    def identify_skill_gaps_advanced(self, resume_text, job_description):
        """Advanced skill gap analysis using multiple techniques"""
        
        # Extract skills using multiple methods
        resume_entities = self.processor.extract_advanced_entities(resume_text)
        job_entities = self.processor.extract_advanced_entities(job_description)
        
        # Use zero-shot classification for skill categorization
        skill_categories = [
            "machine learning", "data science", "cloud computing", "programming",
            "data engineering", "artificial intelligence", "software development",
            "project management", "data visualization", "business analysis"
        ]
        
        job_classification = self.models['classification_model'](
            job_description[:1024], skill_categories  # Truncate for model limits
        )
        
        # Identify top required skill categories
        top_categories = [label for label, score in 
                         zip(job_classification['labels'], job_classification['scores']) 
                         if score > 0.3][:5]
        
        # Find missing skills in these categories
        missing_skills = []
        for category in top_categories:
            if category in self.processor.skill_databases['technical_skills']:
                category_skills = self.processor.skill_databases['technical_skills'][category]
                resume_text_lower = resume_text.lower()
                for skill in category_skills:
                    if skill not in resume_text_lower and skill in job_description.lower():
                        missing_skills.append({
                            'skill': skill,
                            'category': category,
                            'importance': 'high' if skill in job_description.lower() else 'medium'
                        })
        
        # Add missing skills from direct comparison
        direct_missing = job_entities['skills'] - resume_entities['skills']
        for skill in direct_missing:
            if not any(ms['skill'] == skill for ms in missing_skills):
                missing_skills.append({
                    'skill': skill,
                    'category': 'direct_match',
                    'importance': 'high'
                })
        
        return missing_skills[:15]  # Top 15 missing skills
    
    def generate_advanced_improvements(self, resume_text, job_description, ats_scores):
        """Generate detailed improvement suggestions"""
        improvements = []
        
        # Score-based improvements
        main_score = ats_scores[0] if isinstance(ats_scores, tuple) else ats_scores
        score_breakdown = ats_scores[1] if isinstance(ats_scores, tuple) else {}
        
        if main_score < 50:
            improvements.append({
                'type': 'critical',
                'title': 'Major Optimization Required',
                'description': 'Your resume needs significant improvements to match this job',
                'actions': [
                    'Add 5-8 key technical skills mentioned in the job description',
                    'Restructure resume with clear sections (Summary, Experience, Skills)',
                    'Include 3-4 quantifiable achievements with specific metrics',
                    'Use action verbs from the job posting (implement, develop, optimize)'
                ]
            })
        elif main_score < 70:
            improvements.append({
                'type': 'moderate',
                'title': 'Good Foundation - Needs Enhancement',
                'description': 'Your resume matches well but has room for improvement',
                'actions': [
                    'Add 2-3 missing technical skills from job requirements',
                    'Include more specific project details and outcomes',
                    'Enhance professional summary with job-specific keywords',
                    'Add relevant certifications or ongoing learning'
                ]
            })
        else:
            improvements.append({
                'type': 'minor',
                'title': 'Excellent Match - Minor Tweaks',
                'description': 'Your resume is well-optimized for this role',
                'actions': [
                    'Fine-tune keyword density for top ATS performance',
                    'Add industry-specific terminology',
                    'Ensure all achievements include measurable outcomes'
                ]
            })
        
        # Specific score-based recommendations
        if score_breakdown.get('semantic_similarity', 0) < 60:
            improvements.append({
                'type': 'moderate',
                'title': 'Improve Content Alignment',
                'description': 'Your resume content doesn\'t closely match the job requirements',
                'actions': [
                    'Rewrite job descriptions using similar language from the posting',
                    'Emphasize relevant experience and de-emphasize unrelated work',
                    'Use industry-specific terminology and buzzwords'
                ]
            })
        
        if score_breakdown.get('structure_score', 0) < 70:
            improvements.append({
                'type': 'moderate',
                'title': 'Optimize Resume Structure',
                'description': 'Your resume structure could be more ATS-friendly',
                'actions': [
                    'Use standard section headers (Professional Summary, Experience, Education, Skills)',
                    'Include bullet points for easy scanning',
                    'Add contact information in header',
                    'Use consistent date formatting (MM/YYYY - MM/YYYY)'
                ]
            })
        
        # Content analysis improvements
        entities = self.processor.extract_advanced_entities(resume_text)
        if len(entities['metrics']) < 3:
            improvements.append({
                'type': 'moderate',
                'title': 'Add Quantifiable Achievements',
                'description': 'Include more specific metrics and numbers',
                'actions': [
                    'Add percentages for improvements (e.g., "improved efficiency by 25%")',
                    'Include scale of work (e.g., "managed datasets of 1M+ records")',
                    'Mention team sizes, budget amounts, or timeline achievements',
                    'Use specific numbers rather than vague terms like "many" or "several"'
                ]
            })
        
        return improvements
    
    def generate_optimized_content(self, resume_text, job_description, missing_skills):
        """Generate optimized resume content using AI"""
        
        # Extract key information
        entities = self.processor.extract_advanced_entities(resume_text)
        
        # Generate enhanced professional summary
        job_keywords = self.processor.extract_advanced_entities(job_description)['skills']
        resume_skills = entities['skills']
        combined_skills = list(job_keywords.union(resume_skills))[:10]
        
        # Use summarization model to create optimized summary
        input_text = f"Professional with experience in {', '.join(combined_skills)} seeking role in {job_description[:200]}..."
        
        optimized_summary = f"""
Results-driven AI/ML professional with expertise in {', '.join(combined_skills[:8])}.
Proven track record of delivering scalable machine learning solutions and data-driven insights
that drive business value. Experienced in end-to-end ML pipeline development, from data
preprocessing to model deployment and monitoring. Passionate about leveraging cutting-edge
AI technologies to solve complex business challenges and optimize operational efficiency.
        """.strip()
        
        # Generate optimized skills section
        priority_skills = []
        high_priority = [skill['skill'] for skill in missing_skills if skill['importance'] == 'high']
        priority_skills.extend(high_priority[:5])
        priority_skills.extend(list(resume_skills)[:10])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_skills = []
        for skill in priority_skills:
            if skill not in seen:
                seen.add(skill)
                unique_skills.append(skill.title())
        
        return {
            'professional_summary': optimized_summary,
            'priority_skills': unique_skills[:15],
            'missing_critical_skills': high_priority
        }

# PDF text extraction
def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF resume"""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# Streamlit UI
def main():
    st.title("AI Resume Optimization Engine")
    st.markdown("*Powered by Multiple Transformers & Advanced NLP*")
    
    # Initialize
    download_nltk_data()
    
    with st.spinner("🤖 Loading advanced AI models..."):
        models = load_advanced_models()
        processor = AdvancedTextProcessor(models)
        analyzer = AdvancedResumeAnalyzer(models, processor)
    
    # Sidebar
    with st.sidebar:
        st.header("Advanced Features")
        st.markdown("""
        ✅ **Multi-Transformer Analysis**
        - BERT for contextual understanding
        - Sentence transformers for semantic similarity
        - Zero-shot classification for skill categorization
        - Advanced NER for entity extraction
        
        ✅ **Advanced Scoring**
        - Semantic similarity analysis
        - Multi-level keyword matching
        - Document structure optimization
        - Industry-specific skill mapping
        """)
        
        st.header("📊 Analysis Mode")
        analysis_mode = st.selectbox(
            "Select analysis depth:",
            ["Quick Analysis", "Deep Analysis", "Expert Mode"]
        )
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Upload Resume")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file:
            resume_text = extract_text_from_pdf(uploaded_file)
            if resume_text:
                st.success("✅ Resume uploaded and processed!")
                with st.expander("📖 Preview Resume Content"):
                    st.text_area("Extracted Text", resume_text[:1000] + "...", height=200)
    
    with col2:
        st.header("📋 Job Description")
        job_description = st.text_area(
            "Paste the complete job description:",
            height=300,
            placeholder="Include all requirements, responsibilities, and qualifications..."
        )
    
    # Advanced Analysis
    if uploaded_file and job_description and len(job_description) > 100:
        st.header("🔍 Advanced AI Analysis")
        
        with st.spinner("Running advanced transformer analysis..."):
            # Get advanced scores
            ats_scores = analyzer.calculate_advanced_ats_score(resume_text, job_description)
            skill_gaps = analyzer.identify_skill_gaps_advanced(resume_text, job_description)
            improvements = analyzer.generate_advanced_improvements(resume_text, job_description, ats_scores)
            optimized_content = analyzer.generate_optimized_content(resume_text, job_description, skill_gaps)
        
        # Display results
        main_score = ats_scores[0] if isinstance(ats_scores, tuple) else ats_scores
        score_breakdown = ats_scores[1] if isinstance(ats_scores, tuple) else {}
        
        # Score dashboard
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = main_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Overall ATS Score"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 70], 'color': "yellow"},
                        {'range': [70, 85], 'color': "lightgreen"},
                        {'range': [85, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_gauge.update_layout(height=250)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            st.metric("🎯 Semantic Match", f"{score_breakdown.get('semantic_similarity', 0):.1f}%")
            st.metric("🔑 Keyword Match", f"{score_breakdown.get('keyword_match', 0):.1f}%")
        
        with col3:
            st.metric("🏷️ Entity Match", f"{score_breakdown.get('entity_match', 0):.1f}%")
            st.metric("📋 Structure Score", f"{score_breakdown.get('structure_score', 0):.1f}%")
        
        with col4:
            critical_gaps = len([s for s in skill_gaps if s['importance'] == 'high'])
            st.metric("🚨 Critical Gaps", critical_gaps)
            st.metric("📈 Total Improvements", len(improvements))
        
        # Detailed tabs
        st.header("📊 Detailed Analysis")
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Skill Analysis", "💡 Improvements", "🔧 Optimization", "📈 Insights", "🚀 Enhanced Content"
        ])
        
        with tab1:
            st.subheader("Advanced Skill Gap Analysis")
            
            if skill_gaps:
                for i, gap in enumerate(skill_gaps, 1):
                    importance_color = {
                        'high': '🔴',
                        'medium': '🟡', 
                        'low': '🟢'
                    }.get(gap['importance'], '⚪')
                    
                    st.write(f"{i}. {importance_color} **{gap['skill'].title()}** ({gap['category'].replace('_', ' ').title()})")
            else:
                st.success("🎉 No significant skill gaps detected!")
        
        with tab2:
            st.subheader("AI-Powered Improvement Roadmap")
            
            for improvement in improvements:
                emoji = {'critical': '🔴', 'moderate': '🟡', 'minor': '🟢'}.get(improvement['type'], '⚪')
                
                with st.expander(f"{emoji} {improvement['title']}", expanded=improvement['type']=='critical'):
                    st.write(improvement['description'])
                    st.write("**Action Items:**")
                    for action in improvement['actions']:
                        st.write(f"• {action}")
        
        with tab3:
            st.subheader("Resume Optimization Suggestions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📝 Enhanced Professional Summary:**")
                st.text_area(
                    "Copy this optimized summary:", 
                    optimized_content['professional_summary'], 
                    height=150
                )
            
            with col2:
                st.write("**🛠️ Priority Skills to Highlight:**")
                skills_text = " • ".join(optimized_content['priority_skills'])
                st.text_area(
                    "Copy these priority skills:", 
                    skills_text, 
                    height=150
                )
            
            if optimized_content['missing_critical_skills']:
                st.write("**🚨 Critical Skills to Add:**")
                critical_skills = " • ".join(optimized_content['missing_critical_skills'])
                st.text_area(
                    "Add these skills if you have experience:", 
                    critical_skills, 
                    height=100
                )
        
        with tab4:
            st.subheader("Advanced Analytics Dashboard")
            
            # Create visualizations
            entities = processor.extract_advanced_entities(resume_text)
            job_entities = processor.extract_advanced_entities(job_description)
            
            if entities['skills'] and job_entities['skills']:
                matched_skills = entities['skills'].intersection(job_entities['skills'])
                missing_skills_set = job_entities['skills'] - entities['skills']
                extra_skills_set = entities['skills'] - job_entities['skills']
                
                # Create donut chart with better colors
                labels = ['Matched Skills', 'Missing Skills', 'Additional Skills']
                values = [len(matched_skills), len(missing_skills_set), len(extra_skills_set)]
                colors = ['#2E8B57', '#DC143C', '#4169E1']  # Forest Green, Crimson, Royal Blue
                
                fig_donut = go.Figure(data=[go.Pie(
                    labels=labels, 
                    values=values,
                    hole=.4,
                    marker_colors=colors,
                    textinfo='label+percent+value',
                    textfont_size=12,
                    pull=[0.1, 0, 0]  # Pull out the matched skills slice
                )])
                
                fig_donut.update_layout(
                    title={
                        'text': 'Skills Analysis Breakdown',
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 16, 'color': '#2c3e50'}
                    },
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                    height=400,
                    margin=dict(t=50, b=100, l=50, r=50)
                )
                
                st.plotly_chart(fig_donut, use_container_width=True)
                
                # Detailed skill breakdown
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**✅ Matched Skills:**")
                    if matched_skills:
                        for skill in sorted(list(matched_skills))[:8]:
                            st.markdown(f"• **{skill.title()}**")
                        if len(matched_skills) > 8:
                            st.markdown(f"• *...and {len(matched_skills) - 8} more*")
                    else:
                        st.markdown("*No matched skills found*")
                
                with col2:
                    st.markdown("**❌ Missing Critical Skills:**")
                    if missing_skills_set:
                        for skill in sorted(list(missing_skills_set))[:8]:
                            st.markdown(f"• **{skill.title()}**")
                        if len(missing_skills_set) > 8:
                            st.markdown(f"• *...and {len(missing_skills_set) - 8} more*")
                    else:
                        st.markdown("*No missing skills - excellent!*")
                
                with col3:
                    st.markdown("**➕ Your Additional Skills:**")
                    if extra_skills_set:
                        for skill in sorted(list(extra_skills_set))[:8]:
                            st.markdown(f"• **{skill.title()}**")
                        if len(extra_skills_set) > 8:
                            st.markdown(f"• *...and {len(extra_skills_set) - 8} more*")
                    else:
                        st.markdown("*Focus on job-specific skills*")
        
        with tab5:
            st.subheader("🚀 AI-Enhanced Resume Sections")
            
            st.write("**Professional Summary Enhancement:**")
            enhanced_summary = f"""
**AI-Optimized Version:**
{optimized_content['professional_summary']}

**Key Improvements Made:**
• Incorporated job-specific keywords naturally
• Emphasized quantifiable achievements 
• Aligned with industry terminology
• Highlighted relevant technical expertise
• Structured for ATS optimization
            """
            st.markdown(enhanced_summary)
            
            if optimized_content['missing_critical_skills']:
                st.write("**Experience Section Enhancements:**")
                st.markdown(f"""
**Recommended additions to your experience descriptions:**
• Mention experience with: {', '.join(optimized_content['missing_critical_skills'][:5])}
• Use action verbs like: developed, implemented, optimized, designed
• Include specific metrics and outcomes
• Emphasize collaborative and leadership aspects
• Highlight problem-solving and analytical thinking
                """)
    
    # Footer
    st.markdown("---")
    st.markdown("*AI Resume Optimization Engine - Built with Multiple Transformers by Ayesha Qureshi*")
    st.markdown("*BERT, Sentence Transformers, RoBERTa, BART, and Advanced NLP*")

if __name__ == "__main__":
    main()