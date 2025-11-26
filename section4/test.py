"""
نظام ترتيب السير الذاتية الذكي باستخدام Ollama
يستخدم Gemma للـ LLM و embedding-gemma للـ embeddings
مع Cross-Encoder لإعادة الترتيب الدقيق
"""

import numpy as np
from sentence_transformers import CrossEncoder
from typing import List, Dict, Tuple
import json
from dataclasses import dataclass
import requests
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class JobPosting:
    """معلومات الوظيفة"""
    title: str
    description: str
    requirements: str
    
    def get_full_text(self) -> str:
        """دمج كل معلومات الوظيفة"""
        return f"Job Title: {self.title}\n\nDescription: {self.description}\n\nRequirements: {self.requirements}"

@dataclass
class CV:
    """السيرة الذاتية"""
    id: str
    name: str
    content: str
    email: str = ""
    phone: str = ""

@dataclass
class RankedCV:
    """سيرة ذاتية مع الدرجات"""
    cv: CV
    bi_encoder_score: float
    cross_encoder_score: float
    llm_analysis: str = ""
    final_score: float = 0.0


class OllamaClient:
    """عميل للتواصل مع Ollama"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.embedding_model = "embeddinggemma:300m"  # أو "mxbai-embed-large"
        self.llm_model = "gemma3:1b"
    
    def get_embedding(self, text: str) -> np.ndarray:
        """الحصول على embedding من Ollama"""
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.embedding_model,
            "prompt": text
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            embedding = response.json()["embedding"]
            return np.array(embedding)
        except Exception as e:
            print(f"❌ خطأ في الحصول على embedding: {str(e)}")
            raise
    
    def get_embeddings_batch(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """الحصول على embeddings لعدة نصوص"""
        embeddings = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            if show_progress:
                print(f"  جاري المعالجة: {i+1}/{total}", end='\r')
            
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        
        if show_progress:
            print()  # سطر جديد
        
        return np.array(embeddings)
    
    def generate_text(self, prompt: str, max_tokens: int = 500) -> str:
        """توليد نص باستخدام Gemma"""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"❌ خطأ في توليد النص: {str(e)}")
            return f"خطأ في التحليل: {str(e)}"


class CVRankingSystem:
    """النظام الكامل لترتيب السير الذاتية"""
    
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_llm: bool = True,
        top_k_bi_encoder: int = 20
    ):
        """
        تهيئة النظام
        
        Args:
            ollama_url: عنوان خادم Ollama
            cross_encoder_model: نموذج Cross-Encoder لإعادة الترتيب
            use_llm: استخدام Gemma للتحليل المتقدم
            top_k_bi_encoder: عدد السير الذاتية المبدئية
        """
        print("🔄 تهيئة النظام...")
        
        # تهيئة Ollama Client
        self.ollama = OllamaClient(base_url=ollama_url)
        
        # اختبار الاتصال بـ Ollama
        try:
            test_embedding = self.ollama.get_embedding("test")
            print(f"✅ تم الاتصال بـ Ollama بنجاح! (Embedding model: {self.ollama.embedding_model})")
        except Exception as e:
            print(f"❌ فشل الاتصال بـ Ollama: {str(e)}")
            print("⚠️ تأكد من تشغيل Ollama: ollama serve")
            raise
        
        # تحميل Cross-Encoder
        print("🔄 تحميل Cross-Encoder...")
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        print(f"✅ تم تحميل Cross-Encoder: {cross_encoder_model}")
        
        self.use_llm = use_llm
        self.top_k = top_k_bi_encoder
        
        if self.use_llm:
            print(f"✅ تم تفعيل LLM للتحليل المتقدم (Model: {self.ollama.llm_model})")
        
        print("✅ النظام جاهز للاستخدام!")
    
    def encode_job(self, job: JobPosting) -> np.ndarray:
        """تحويل الوظيفة إلى embedding باستخدام Ollama"""
        job_text = job.get_full_text()
        print("🔄 تحويل معلومات الوظيفة إلى embedding...")
        return self.ollama.get_embedding(job_text)
    
    def encode_cvs(self, cvs: List[CV]) -> np.ndarray:
        """تحويل السير الذاتية إلى embeddings"""
        print(f"🔄 تحويل {len(cvs)} سيرة ذاتية إلى embeddings...")
        cv_texts = [cv.content for cv in cvs]
        return self.ollama.get_embeddings_batch(cv_texts, show_progress=True)
    
    def bi_encoder_search(
        self,
        job_embedding: np.ndarray,
        cv_embeddings: np.ndarray,
        cvs: List[CV]
    ) -> List[Tuple[CV, float]]:
        """
        البحث الأولي باستخدام embeddings (سريع)
        يحسب cosine similarity بين الوظيفة والسير الذاتية
        """
        print(f"\n🔍 المرحلة 1: البحث الأولي باستخدام Embeddings...")
        
        # حساب cosine similarity
        job_embedding_2d = job_embedding.reshape(1, -1)
        similarities = cosine_similarity(job_embedding_2d, cv_embeddings)[0]
        
        # ترتيب النتائج
        top_indices = np.argsort(similarities)[::-1][:self.top_k]
        results = [(cvs[idx], float(similarities[idx])) for idx in top_indices]
        
        print(f"✅ تم اختيار أفضل {len(results)} سيرة ذاتية")
        return results
    
    def cross_encoder_rerank(
        self,
        job: JobPosting,
        candidate_cvs: List[Tuple[CV, float]]
    ) -> List[RankedCV]:
        """
        إعادة الترتيب باستخدام Cross-Encoder (أكثر دقة)
        """
        print(f"\n🎯 المرحلة 2: إعادة الترتيب باستخدام Cross-Encoder...")
        
        job_text = job.get_full_text()
        pairs = [[job_text, cv.content] for cv, _ in candidate_cvs]
        
        # حساب الدرجات باستخدام Cross-Encoder
        print("  جاري حساب الدرجات...")
        cross_scores = self.cross_encoder.predict(pairs)
        
        # إنشاء قائمة مرتبة
        ranked_cvs = []
        for (cv, bi_score), cross_score in zip(candidate_cvs, cross_scores):
            ranked_cv = RankedCV(
                cv=cv,
                bi_encoder_score=bi_score,
                cross_encoder_score=float(cross_score),
                final_score=float(cross_score)
            )
            ranked_cvs.append(ranked_cv)
        
        # ترتيب تنازلي حسب الدرجة النهائية
        ranked_cvs.sort(key=lambda x: x.final_score, reverse=True)
        
        print(f"✅ تم إعادة ترتيب السير الذاتية بدقة عالية")
        return ranked_cvs
    
    def llm_analysis(
        self,
        job: JobPosting,
        ranked_cvs: List[RankedCV],
        top_n: int = 5
    ) -> List[RankedCV]:
        """
        تحليل متقدم باستخدام Gemma للسير الذاتية الأفضل
        """
        if not self.use_llm:
            print("⚠️ تم تخطي تحليل LLM")
            return ranked_cvs
        
        print(f"\n🤖 المرحلة 3: التحليل المتقدم باستخدام Gemma لأفضل {top_n} سير ذاتية...")
        
        for i, ranked_cv in enumerate(ranked_cvs[:top_n]):
            try:
                # تحديد طول محتوى السيرة الذاتية (لتجنب تجاوز الحد)
                cv_content = ranked_cv.cv.content[:1500]
                
                prompt = f"""You are an expert HR recruiter. Analyze how well this CV matches the job posting.

Job Posting:
{job.get_full_text()}

CV:
{cv_content}

Provide a concise analysis (3-4 lines) covering:
1. Main strengths
2. Matching skills
3. Any gaps or weaknesses
4. Overall assessment (Excellent/Very Good/Good/Fair/Poor)

Analysis:"""

                print(f"  🔄 تحليل السيرة الذاتية {i+1}/{top_n}...", end='')
                analysis = self.ollama.generate_text(prompt, max_tokens=300)
                ranked_cv.llm_analysis = analysis.strip()
                print(" ✅")
                
            except Exception as e:
                print(f" ❌")
                print(f"  ⚠️ خطأ في تحليل السيرة {i+1}: {str(e)}")
                ranked_cv.llm_analysis = "لم يتم التحليل بسبب خطأ تقني"
        
        return ranked_cvs
    
    def rank_cvs(
        self,
        job: JobPosting,
        cvs: List[CV],
        use_llm_analysis: bool = None
    ) -> List[RankedCV]:
        """
        الدالة الرئيسية: ترتيب السير الذاتية بالكامل
        
        Args:
            job: معلومات الوظيفة
            cvs: قائمة السير الذاتية
            use_llm_analysis: استخدام LLM للتحليل (None = استخدام الإعداد الافتراضي)
        
        Returns:
            قائمة مرتبة من السير الذاتية مع الدرجات والتحليل
        """
        print("="*70)
        print("🚀 بدء عملية ترتيب السير الذاتية")
        print("="*70)
        
        # 1. Embeddings: البحث السريع
        job_embedding = self.encode_job(job)
        cv_embeddings = self.encode_cvs(cvs)
        candidate_cvs = self.bi_encoder_search(job_embedding, cv_embeddings, cvs)
        
        # 2. Cross-Encoder: إعادة الترتيب الدقيق
        ranked_cvs = self.cross_encoder_rerank(job, candidate_cvs)
        
        # 3. Gemma LLM: التحليل المتقدم (اختياري)
        if use_llm_analysis is None:
            use_llm_analysis = self.use_llm
        
        if use_llm_analysis:
            ranked_cvs = self.llm_analysis(job, ranked_cvs)
        
        print("\n" + "="*70)
        print("✅ اكتملت عملية الترتيب بنجاح!")
        print("="*70)
        
        return ranked_cvs
    
    def print_results(self, ranked_cvs: List[RankedCV], top_n: int = 10):
        """طباعة النتائج بشكل منسق"""
        print(f"\n📊 أفضل {min(top_n, len(ranked_cvs))} سير ذاتية:")
        print("="*80)
        
        for i, ranked_cv in enumerate(ranked_cvs[:top_n], 1):
            print(f"\n🏆 المرتبة {i}:")
            print(f"   الاسم: {ranked_cv.cv.name}")
            print(f"   ID: {ranked_cv.cv.id}")
            if ranked_cv.cv.email:
                print(f"   Email: {ranked_cv.cv.email}")
            print(f"   درجة Embedding: {ranked_cv.bi_encoder_score:.4f}")
            print(f"   درجة Cross-Encoder: {ranked_cv.cross_encoder_score:.4f}")
            print(f"   الدرجة النهائية: {ranked_cv.final_score:.4f}")
            
            if ranked_cv.llm_analysis:
                print(f"\n   📝 تحليل Gemma:")
                for line in ranked_cv.llm_analysis.split('\n'):
                    if line.strip():
                        print(f"      {line.strip()}")
            
            print("-"*80)
    
    def save_results(self, ranked_cvs: List[RankedCV], filename: str = "cv_ranking_results.json"):
        """حفظ النتائج في ملف JSON"""
        results_dict = []
        for rank, ranked_cv in enumerate(ranked_cvs, 1):
            results_dict.append({
                "rank": rank,
                "name": ranked_cv.cv.name,
                "id": ranked_cv.cv.id,
                "email": ranked_cv.cv.email,
                "phone": ranked_cv.cv.phone,
                "embedding_score": round(ranked_cv.bi_encoder_score, 4),
                "cross_encoder_score": round(ranked_cv.cross_encoder_score, 4),
                "final_score": round(ranked_cv.final_score, 4),
                "llm_analysis": ranked_cv.llm_analysis
            })
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 تم حفظ النتائج في: {filename}")


# =============================================================================
# مثال على الاستخدام
# =============================================================================

def main():
    """مثال عملي على استخدام النظام"""
    
    # 1. إنشاء معلومات الوظيفة
    job = JobPosting(
        title="Senior Python Developer",
        description="""
        We are looking for an experienced Python developer to join our team.
        You will work on building scalable web applications using Django and FastAPI.
        Experience with machine learning and NLP is a plus.
        """,
        requirements="""
        - 5+ years of Python development experience
        - Strong knowledge of Django/FastAPI
        - Experience with PostgreSQL and Redis
        - Familiarity with Docker and Kubernetes
        - Good understanding of ML/NLP concepts
        - Excellent problem-solving skills
        """
    )
    
    # 2. إنشاء سير ذاتية تجريبية
    cvs = [
        CV(
            id="CV001",
            name="Ahmed Mohamed",
            content="""
            Senior Software Engineer with 6 years of experience in Python development.
            Expert in Django, FastAPI, and RESTful APIs. Built multiple scalable web applications
            serving millions of users. Proficient in PostgreSQL, Redis, and MongoDB.
            Experience with Docker, Kubernetes, and AWS. Strong background in machine learning
            and natural language processing. Led team of 5 developers in previous role.
            Skills: Python, Django, FastAPI, PostgreSQL, Redis, Docker, Kubernetes, ML, NLP, AWS
            """,
            email="ahmed@example.com",
            phone="+20 123 456 7890"
        ),
        CV(
            id="CV002",
            name="Sara Ali",
            content="""
            Python Developer with 3 years of experience. Worked primarily with Flask and Django.
            Good knowledge of SQL databases and basic understanding of Docker.
            Recent graduate with Master's degree in Computer Science.
            Interested in learning more about cloud technologies.
            Skills: Python, Django, Flask, MySQL, Git, Basic Docker
            """,
            email="sara@example.com",
            phone="+20 100 111 2222"
        ),
        CV(
            id="CV003",
            name="Omar Hassan",
            content="""
            Full Stack Developer with 7 years experience. Expert in Python, JavaScript, React.
            Built enterprise applications using Django and Node.js. Strong DevOps skills
            including Docker, Kubernetes, CI/CD pipelines. Experience with ML model deployment
            and MLOps. Contributed to several open-source NLP projects. AWS Certified Solutions Architect.
            Skills: Python, Django, FastAPI, React, Node.js, PostgreSQL, Redis, Docker, Kubernetes, ML, NLP, AWS, MLOps
            """,
            email="omar@example.com",
            phone="+20 122 333 4444"
        ),
        CV(
            id="CV004",
            name="Fatima Khalil",
            content="""
            Junior Python Developer with 1 year of experience. Knowledge of basic Python
            and Flask framework. Completed online courses in web development.
            Eager to learn and grow in the field. Strong academic background.
            Skills: Python, Flask, HTML, CSS, JavaScript, Git
            """,
            email="fatima@example.com",
            phone="+20 111 222 3333"
        ),
        CV(
            id="CV005",
            name="Khaled Ibrahim",
            content="""
            AI/ML Engineer with 5 years experience specializing in NLP and deep learning.
            Proficient in Python, TensorFlow, PyTorch, and scikit-learn. Built chatbots
            and text classification systems. Good knowledge of FastAPI for model serving.
            Experience with Docker and basic DevOps. Published research papers in NLP.
            Skills: Python, TensorFlow, PyTorch, scikit-learn, FastAPI, NLP, Deep Learning, Docker
            """,
            email="khaled@example.com",
            phone="+20 155 666 7777"
        ),
        CV(
            id="CV006",
            name="Layla Hussein",
            content="""
            Backend Developer with 4 years experience. Specialized in Python and Go.
            Built microservices architecture using FastAPI and gRPC. Good experience with
            distributed systems and message queues (RabbitMQ, Kafka). Knowledge of PostgreSQL,
            MongoDB, and Redis. Familiar with Docker and Kubernetes.
            Skills: Python, Go, FastAPI, gRPC, PostgreSQL, MongoDB, Redis, Docker, Kubernetes, Microservices
            """,
            email="layla@example.com",
            phone="+20 101 888 9999"
        )
    ]
    
    # 3. تهيئة النظام
    # تأكد من تشغيل Ollama أولاً: ollama serve
    # وتحميل النماذج: ollama pull gemma2 && ollama pull nomic-embed-text
    
    try:
        system = CVRankingSystem(
            ollama_url="http://localhost:11434",
            use_llm=True,  # تفعيل Gemma للتحليل المتقدم
            top_k_bi_encoder=10  # عدد السير المبدئية
        )
        
        # 4. ترتيب السير الذاتية
        ranked_results = system.rank_cvs(job, cvs, use_llm_analysis=True)
        
        # 5. عرض النتائج
        system.print_results(ranked_results, top_n=6)
        
        # 6. حفظ النتائج
        system.save_results(ranked_results)
        
    except Exception as e:
        print(f"\n❌ حدث خطأ: {str(e)}")
        print("\n💡 تأكد من:")
        print("   1. تشغيل Ollama: ollama serve")
        print("   2. تحميل النماذج:")
        print("      - ollama pull gemma2")
        print("      - ollama pull nomic-embed-text")


if __name__ == "__main__":
    main()