import os
import time
import json
import re
import io
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import streamlit as st
import PyPDF2
import docx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Optional: Google Trends via pytrends (no API key needed) ---
HAS_PYTRENDS = True
try:
    from pytrends.request import TrendReq
except Exception:
    HAS_PYTRENDS = False

# ---------------------- Enhanced dictionaries for fresh graduates ----------------------
CAREER_PATHS = {
    'Software Engineer':  {
        'progression': ['Senior Software Engineer', 'Tech Lead', 'Engineering Manager', 'Principal Engineer'],
        'entry_keywords': ['web development', 'frontend', 'backend', 'full stack', 'programming', 'coding'],
        'fresh_grad_roles': ['Junior Developer', 'Frontend Developer', 'Backend Developer', 'Full Stack Developer']
    },
    'Data Scientist':     {
        'progression': ['Senior Data Scientist', 'Lead Data Scientist', 'ML Engineering Manager', 'Chief Data Officer'],
        'entry_keywords': ['data analysis', 'machine learning', 'statistics', 'analytics', 'data mining'],
        'fresh_grad_roles': ['Data Analyst', 'Junior Data Scientist', 'ML Engineer', 'Business Intelligence Analyst']
    },
    'DevOps Engineer':    {
        'progression': ['Senior DevOps Engineer', 'Platform Engineer', 'DevOps Manager', 'VP Infrastructure'],
        'entry_keywords': ['deployment', 'automation', 'ci/cd', 'infrastructure', 'cloud'],
        'fresh_grad_roles': ['Junior DevOps Engineer', 'Infrastructure Engineer', 'Build Engineer', 'Site Reliability Engineer']
    },
    'Cybersecurity Specialist': {
        'progression': ['Senior Security Engineer', 'Security Architect', 'Security Manager', 'CISO'],
        'entry_keywords': ['security', 'ethical hacking', 'penetration testing', 'vulnerability', 'cybersecurity'],
        'fresh_grad_roles': ['Security Analyst', 'Junior Security Engineer', 'SOC Analyst', 'Incident Response Analyst']
    },
    'Cloud Engineer':     {
        'progression': ['Senior Cloud Engineer', 'Cloud Solutions Architect', 'Cloud Engineering Manager', 'VP Cloud'],
        'entry_keywords': ['cloud computing', 'aws', 'azure', 'gcp', 'cloud infrastructure'],
        'fresh_grad_roles': ['Junior Cloud Engineer', 'Cloud Support Engineer', 'Infrastructure Engineer', 'Cloud Architect']
    },
    'Product Manager':    {
        'progression': ['Senior Product Manager', 'Group PM', 'Director of Product', 'VP Product'],
        'entry_keywords': ['product management', 'user experience', 'business analysis', 'market research'],
        'fresh_grad_roles': ['Associate Product Manager', 'Product Analyst', 'Business Analyst', 'Product Coordinator']
    },
    'UI/UX Designer':     {
        'progression': ['Senior UX Designer', 'Design Lead', 'Design Manager', 'VP Design'],
        'entry_keywords': ['design', 'user interface', 'user experience', 'wireframes', 'prototyping'],
        'fresh_grad_roles': ['Junior UX Designer', 'UI Designer', 'Product Designer', 'Interaction Designer']
    },
    'Business Analyst':   {
        'progression': ['Senior Business Analyst', 'Lead Business Analyst', 'Product Manager', 'Director Strategy'],
        'entry_keywords': ['business analysis', 'requirements', 'process improvement', 'documentation'],
        'fresh_grad_roles': ['Junior Business Analyst', 'Systems Analyst', 'Process Analyst', 'Data Analyst']
    },
    'Marketing Specialist':{
        'progression': ['Marketing Manager', 'Senior Marketing Manager', 'VP Marketing', 'CMO'],
        'entry_keywords': ['marketing', 'social media', 'content', 'branding', 'digital marketing'],
        'fresh_grad_roles': ['Marketing Coordinator', 'Digital Marketing Executive', 'Content Creator', 'Social Media Manager']
    },
    'Sales Representative':{
        'progression': ['Senior Sales Rep', 'Sales Manager', 'Regional Sales Director', 'VP Sales'],
        'entry_keywords': ['sales', 'customer service', 'lead generation', 'business development'],
        'fresh_grad_roles': ['Sales Associate', 'Business Development Associate', 'Account Executive', 'Inside Sales Rep']
    }
}

SKILL_KEYWORDS = {
    'Software Engineer': {
        'core': ['python', 'java', 'javascript', 'react', 'html', 'css', 'sql', 'git'],
        'advanced': ['microservices', 'docker', 'kubernetes', 'api', 'mongodb', 'postgresql', 'django'],
        'trending': ['rust', 'go', 'typescript', 'graphql', 'serverless', 'flutter', 'react.js']
    },
    'Data Scientist': {
        'core': ['python', 'r', 'sql', 'pandas', 'numpy', 'scikit-learn', 'statistics'],
        'advanced': ['tensorflow', 'pytorch', 'keras', 'spark', 'hadoop', 'tableau'],
        'trending': ['llm', 'transformers', 'mlops', 'mlflow', 'dvc', 'kubeflow']
    },
    'DevOps Engineer': {
        'core': ['docker', 'kubernetes', 'linux', 'bash', 'git', 'ci', 'cd', 'ci/cd'],
        'advanced': ['terraform', 'ansible', 'jenkins', 'aws', 'azure', 'monitoring'],
        'trending': ['gitops', 'helm', 'istio', 'prometheus', 'grafana', 'argo']
    },
    'Cybersecurity Specialist': {
        'core': ['security', 'penetration testing', 'vulnerability', 'firewall', 'encryption', 'ethical hacking'],
        'advanced': ['siem', 'incident response', 'compliance', 'risk assessment', 'forensics'],
        'trending': ['zero trust', 'devsecops', 'cloud security', 'ai security', 'quantum cryptography']
    },
    'Cloud Engineer': {
        'core': ['aws', 'azure', 'gcp', 'cloud architecture', 'networking'],
        'advanced': ['serverless', 'lambda', 'cloudformation', 'terraform', 'containers'],
        'trending': ['multi-cloud', 'edge computing', 'cloud native', 'service mesh', 'finops']
    },
    'Product Manager': {
        'core': ['product management', 'agile', 'scrum', 'roadmap', 'user stories'],
        'advanced': ['analytics', 'a/b testing', 'market research', 'competitive analysis'],
        'trending': ['ai product management', 'data-driven decisions', 'growth hacking', 'ux research']
    },
    'UI/UX Designer': {
        'core': ['figma', 'sketch', 'adobe', 'user experience', 'user interface', 'wireframes'],
        'advanced': ['prototyping', 'user research', 'information architecture', 'interaction design'],
        'trending': ['design systems', 'accessibility', 'voice ui', 'ar/vr design', 'micro-interactions']
    },
    'Business Analyst': {
        'core': ['business analysis', 'requirements', 'process improvement', 'documentation'],
        'advanced': ['sql', 'tableau', 'power bi', 'stakeholder management', 'gap analysis'],
        'trending': ['data analysis', 'automation', 'digital transformation', 'customer journey mapping']
    },
    'Marketing Specialist': {
        'core': ['digital marketing', 'seo', 'social media', 'content marketing', 'email marketing'],
        'advanced': ['google analytics', 'facebook ads', 'marketing automation', 'crm'],
        'trending': ['growth marketing', 'influencer marketing', 'marketing ai', 'programmatic advertising']
    },
    'Sales Representative': {
        'core': ['sales', 'crm', 'lead generation', 'negotiation', 'customer relationship'],
        'advanced': ['salesforce', 'pipeline management', 'territory management', 'account management'],
        'trending': ['social selling', 'sales automation', 'predictive analytics', 'video selling']
    }
}

# ---------------------- Enhanced Resume Processing for Fresh Graduates ----------------------
class AdvancedResumeProcessor:
    def __init__(self):
        self.resume_indicators = [
            'experience', 'education', 'skills', 'work', 'employment', 'projects',
            'achievements', 'qualifications', 'resume', 'cv', 'curriculum vitae',
            'objective', 'summary', 'profile', 'internship', 'coursework'
        ]

    @staticmethod
    def extract_text_from_pdf(file) -> str:
        try:
            file.seek(0)
            pdf = PyPDF2.PdfReader(file)
            text = ""
            for p in pdf.pages:
                t = p.extract_text() or ""
                text += t + "\n"
            return text.strip()
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""

    @staticmethod
    def extract_text_from_docx(file) -> str:
        try:
            file.seek(0)
            doc = docx.Document(file)
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            return text.strip()
        except Exception as e:
            st.error(f"Error reading DOCX: {e}")
            return ""

    def is_resume(self, text: str) -> bool:
        tl = text.lower()
        indicator_score = sum(2 for ind in self.resume_indicators if ind in tl)
       
        structure_score = 0
        structure_score += 5 if re.search(r'\b\w+@\w+\.\w+\b', tl) else 0
        structure_score += 4 if re.search(r'\b(phone|mobile|\+?\d{1,3}[-.\s]?\d{5,})', tl) else 0
        structure_score += 6 if any(k in tl for k in ['experience', 'work history', 'employment', 'internship']) else 0
        structure_score += 4 if 'education' in tl else 0
        structure_score += 3 if 'skills' in tl else 0
        structure_score += 3 if 'projects' in tl else 0
       
        pattern_score = 0
        pattern_score += 3 if re.search(r'\b(19|20)\d{2}\s*[-â€"â€"]\s*(19|20)\d{2}\b', tl) else 0
        pattern_score += 3 if re.search(r'\b(19|20)\d{2}\s*[-â€"â€"]\s*present\b', tl) else 0
       
        length_score = 5 if 50 <= len(text.split()) <= 3000 else 0
        total = indicator_score + structure_score + pattern_score + length_score
       
        return total >= 12  # Lower threshold for student resumes

    def clean(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\-\@\(\)\+\#\&/]', ' ', text)
        return text.strip()

    def extract_features(self, text: str) -> Dict:
        tl = text.lower()
       
        # Enhanced experience extraction for fresh graduates
        years = 0
       
        # Look for internship experience first
        internship_patterns = [
            r'intern(?:ship)?\s*(?:experience)?.*?(\d+)\s*months?',
            r'(\d+)\s*months?\s*intern(?:ship)?',
            r'intern.*?(\d{4})\s*[-â€"]\s*(\d{4})',
            r'intern.*?(\d{4})\s*[-â€"]\s*present'
        ]
       
        current_year = datetime.now().year
        for pattern in internship_patterns:
            matches = re.findall(pattern, tl)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        if len(match) == 2 and match[0].isdigit():
                            if len(match[0]) == 4:  # Year format
                                start_year = int(match[0])
                                if 'present' in match[1].lower():
                                    years += max(0, (current_year - start_year) * 0.8)  # Internship weight
                                elif match[1].isdigit() and len(match[1]) == 4:
                                    end_year = int(match[1])
                                    years += max(0, (end_year - start_year) * 0.8)
                    else:
                        if match.isdigit():
                            months = int(match)
                            years += months / 12.0
       
        # Look for project experience if no work experience
        if years == 0:
            project_count = len(re.findall(r'project', tl))
            if project_count > 0:
                years = min(project_count * 0.3, 2)  # Projects contribute to experience
       
        # Academic year calculation for current students
        academic_patterns = [
            r'(\d{4})\s*[-â€"]\s*present',
            r'(\d{4})\s*[-â€"]\s*(\d{4})'
        ]
       
        for pattern in academic_patterns:
            matches = re.findall(pattern, tl)
            if matches:
                for match in matches:
                    if isinstance(match, tuple) and len(match) >= 1:
                        start_year = int(match[0])
                        if len(match) == 2 and match[1].isdigit():
                            end_year = int(match[1])
                        else:
                            end_year = current_year
                       
                        # If it's likely an education period, don't count as work experience
                        # but use it to estimate student status
                        if end_year - start_year <= 5 and start_year >= current_year - 10:
                            # This is likely educational experience
                            pass

        # Enhanced education level detection
        edu_map = {
            'phd': 5, 'ph.d': 5, 'doctorate': 5, 'doctoral': 5,
            'master': 4, 'm.s': 4, 'mtech': 4, 'm.tech': 4, 'mba': 4, 'ms': 4, 'm.sc': 4,
            'bachelor': 3, 'b.tech': 3, 'b.e': 3, 'bs': 3, 'b.s': 3, 'be': 3, 'btech': 3, 'b.sc': 3,
            'diploma': 2, 'associate': 2,
            'certificate': 1, 'certification': 1,
            'higher secondary': 2, 'high school': 1, '12th': 2, 'intermediate': 2
        }
       
        edu_level = 0
        edu_details = []

# Force word boundaries to avoid "ms" in emails etc.
        for keyword, level in edu_map.items():
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, tl):
        # Only keep the HIGHEST qualification found
                if level > edu_level:
                    edu_level = level
                edu_details.append(keyword)

# If multiple levels are found, keep only the top one
        if edu_details:
            highest = max(edu_details, key=lambda k: edu_map[k])
            edu_details = [highest]

       
        # Enhanced skills detection with better matching
        skill_scores = {}
        all_skills = []
       
        for career, categories in SKILL_KEYWORDS.items():
            for category, skills in categories.items():
                weight = {'core': 3, 'advanced': 2, 'trending': 4}.get(category, 1)
                for skill in skills:
                    # More flexible skill matching
                    patterns = [
                        r'\b' + re.escape(skill.lower()) + r'\b',
                        r'\b' + re.escape(skill.lower().replace('.', '\.')) + r'\b',
                        r'\b' + re.escape(skill.lower().replace('-', '[-\s]?')) + r'\b'
                    ]
                   
                    for pattern in patterns:
                        if re.search(pattern, tl):
                            all_skills.append(skill)
                            skill_scores[skill] = skill_scores.get(skill, 0) + weight
                            break
       
        # Career relevance scoring with fresh graduate considerations
        career_scores = {}
        for career, career_data in CAREER_PATHS.items():
            score = 0
           
            # Skills-based scoring
            career_skills = SKILL_KEYWORDS.get(career, {})
            for category, skills in career_skills.items():
                category_weight = {'core': 1.5, 'advanced': 1.2, 'trending': 1.8}.get(category, 1.0)
                for skill in skills:
                    if skill in [s.lower() for s in all_skills]:
                        skill_score = skill_scores.get(skill, 1)
                        score += skill_score * category_weight
           
            # Entry-level keyword bonus
            entry_keywords = career_data.get('entry_keywords', [])
            for keyword in entry_keywords:
                if keyword in tl:
                    score += 5  # Bonus for entry-level relevant terms
           
            # Project relevance bonus
            project_bonus = 0
            if 'projects' in tl:
                project_sections = re.findall(r'projects?\s*:?(.*?)(?=\n\s*[A-Z]|$)', tl, re.IGNORECASE | re.DOTALL)
                for section in project_sections:
                    for skill in career_skills.get('core', []) + career_skills.get('advanced', []):
                        if skill in section.lower():
                            project_bonus += 2
           
            career_scores[career] = score + project_bonus
       
        # Enhanced seniority indicators (adjusted for students/fresh graduates)
        senior_terms = {
            'leadership': ['lead', 'led', 'leading', 'managed', 'supervised', 'coordinated', 'team leader'],
            'management': ['manager', 'management', 'director', 'head', 'president', 'coordinator'],
            'mentoring': ['mentor', 'mentoring', 'coached', 'training', 'taught', 'guided', 'tutoring'],
            'technical': ['developed', 'built', 'created', 'designed', 'implemented', 'architected'],
            'achievement': ['achieved', 'accomplished', 'awarded', 'recognized', 'certified', 'solved']
        }
       
        seniority = {}
        for category, terms in senior_terms.items():
            count = sum(1 for term in terms if re.search(r'\b' + re.escape(term) + r'\b', tl))
            seniority[category] = count
       
        # Coursework and certification detection
        coursework = []
        coursework_patterns = [
            r'coursework[:\s]*(.*?)(?=\n[A-Z]|\n\s*$|$)',
            r'relevant courses?[:\s]*(.*?)(?=\n[A-Z]|\n\s*$|$)',
            r'subjects?[:\s]*(.*?)(?=\n[A-Z]|\n\s*$|$)'
        ]
       
        for pattern in coursework_patterns:
            matches = re.findall(pattern, tl, re.IGNORECASE | re.DOTALL)
            for match in matches:
                courses = [course.strip() for course in re.split(r'[,â€¢\n]', match) if course.strip()]
                coursework.extend(courses[:10])  # Limit coursework items
       
        return {
            'years_experience': min(years, 10),  # Cap at 10 for fresh graduates
            'education_level': edu_level,
            'education_details': edu_details[:6],
            'skills': sorted(set(all_skills)),
            'skill_scores': skill_scores,
            'career_scores': career_scores,
            'seniority_indicators': seniority,
            'coursework': coursework[:10],
            'is_fresh_graduate': years < 2 and edu_level >= 3  # Bachelor's degree with <2 years experience
        }

# ---------------------- Lightweight TTL cache ----------------------
class TTLCache:
    def __init__(self, ttl_seconds=21600):
        self.ttl = ttl_seconds
        self.store: Dict[str, Tuple[float, dict]] = {}
   
    def get(self, key: str):
        now = time.time()
        if key in self.store:
            t, val = self.store[key]
            if now - t < self.ttl:
                return val
            else:
                del self.store[key]
        return None
   
    def set(self, key: str, val: dict):
        self.store[key] = (time.time(), val)

rt_cache = TTLCache(ttl_seconds=6*3600)

# ---------------------- Real-time signals provider ----------------------
class RealTimeSignals:
    def __init__(self, github_token: str | None = None, location_hint: str = "India"):
        self.gh_token = github_token
        self.location_hint = location_hint

    def _headers(self):
        h = {'Accept': 'application/vnd.github+json'}
        if self.gh_token:
            h['Authorization'] = f'Bearer {self.gh_token}'
        return h

    def stackoverflow_tags(self, tags: List[str]) -> Dict[str, dict]:
        tags = [t.lower().replace(' ', '-') for t in tags]
        key = f"so:{','.join(sorted(tags))}"
        cached = rt_cache.get(key)
        if cached: return cached
       
        out = {}
        # Batch up to 20 tags per call
        for i in range(0, len(tags), 20):
            chunk = tags[i:i+20]
            url = f"https://api.stackexchange.com/2.3/tags/{';'.join(chunk)}/info?site=stackoverflow"
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200:
                    data = r.json().get('items', [])
                    for item in data:
                        out[item['name']] = {
                            'count': item.get('count', 0),
                            'has_synonyms': item.get('has_synonyms', False)
                        }
            except Exception:
                continue
       
        out['_fetched_at'] = datetime.utcnow().isoformat()+'Z'
        out['_source'] = 'StackExchange API'
        rt_cache.set(key, out)
        return out

    def github_search(self, terms: List[str], since_days: int = 180) -> Dict[str, dict]:
        key = f"gh:{','.join(sorted(terms))}:{since_days}"
        cached = rt_cache.get(key)
        if cached: return cached
       
        out = {}
        since_date = (datetime.utcnow() - timedelta(days=since_days)).date().isoformat()
       
        for term in terms:
            q = f"{term} in:name,description,readme pushed:>={since_date}"
            try:
                r = requests.get(
                    "https://api.github.com/search/repositories",
                    params={'q': q, 'per_page': 1},
                    headers=self._headers(),
                    timeout=15
                )
                if r.status_code == 200:
                    j = r.json()
                    out[term.lower()] = {'repos_last_6m': int(j.get('total_count', 0))}
                else:
                    out[term.lower()] = {'repos_last_6m': 0}
            except Exception:
                out[term.lower()] = {'repos_last_6m': 0}
       
        out['_fetched_at'] = datetime.utcnow().isoformat()+'Z'
        out['_source'] = 'GitHub Search API'
        rt_cache.set(key, out)
        return out

    def google_trends(self, terms: List[str]) -> Dict[str, float]:
        if not HAS_PYTRENDS or not terms:
            return {}
       
        key = f"gt:{','.join(sorted(terms))}"
        cached = rt_cache.get(key)
        if cached: return cached
       
        out = {}
        try:
            pytrends = TrendReq(hl='en-US', tz=0)
            # Google Trends allows up to 5 terms per request; batch if needed
            for i in range(0, len(terms), 5):
                chunk = terms[i:i+5]
                pytrends.build_payload(chunk, timeframe='today 3-m', geo='')
                df = pytrends.interest_over_time()
                if df is not None and not df.empty:
                    latest = df.tail(4).mean()  # average of ~last month
                    for t in chunk:
                        out[t.lower()] = float(latest.get(t, 0.0))
        except Exception:
            pass
       
        out['_fetched_at'] = datetime.utcnow().isoformat()+'Z'
        out['_source'] = 'Google Trends (pytrends)'
        rt_cache.set(key, out)
        return out

    def career_demand_snapshot(self, features: Dict) -> Dict[str, dict]:
        # Build candidate skill set and per-career relevant skill sets
        candidate_skills = [s.lower() for s in features.get('skills', [])]
       
        # To ensure coverage, include top keywords per career
        per_career_terms = {}
        for career, cats in SKILL_KEYWORDS.items():
            terms = []
            for cat, skills in cats.items():
                terms.extend(skills[:6])  # top few skills per category
            # Boost intersection with candidate skills
            inter = [t for t in terms if t.lower() in candidate_skills]
            terms = list(dict.fromkeys(inter + terms))[:10]
            per_career_terms[career] = terms

        # Merge all unique terms for API calls
        all_terms = sorted(set([t for ts in per_career_terms.values() for t in ts]))

        so = self.stackoverflow_tags(all_terms)
        gh = self.github_search(all_terms)
        gt = self.google_trends(all_terms)

        # Compute normalized demand index per career
        def nz_max(values): return max(values) if values else 1.0
        so_max = nz_max([v['count'] for k,v in so.items() if not k.startswith('_') and 'count' in v])
        gh_max = nz_max([v['repos_last_6m'] for k,v in gh.items() if not k.startswith('_') and 'repos_last_6m' in v])
        gt_max = nz_max([v for k,v in gt.items() if not k.startswith('_')])

        snapshot = {}
        for career, terms in per_career_terms.items():
            s_sum = sum(so.get(t.lower(), {}).get('count', 0) for t in terms)
            g_sum = sum(gh.get(t.lower(), {}).get('repos_last_6m', 0) for t in terms)
            t_sum = sum(gt.get(t.lower(), 0.0) for t in terms)

            s_norm = (s_sum / (so_max * len(terms))) if so_max else 0.0
            g_norm = (g_sum / (gh_max * len(terms))) if gh_max else 0.0
            t_norm = (t_sum / (gt_max * len(terms))) if gt_max else 0.0

            demand_index = 0.5 * s_norm + 0.35 * g_norm + 0.15 * t_norm  # weighted blend
            snapshot[career] = {
                'demand_index': float(min(1.0, demand_index)),
                'signals': {
                    'stack_overflow_total': int(s_sum),
                    'github_repos_6m_total': int(g_sum),
                    'google_trends_sum': float(t_sum)
                },
                'top_terms': terms[:8]
            }

        snapshot['_sources'] = [so.get('_source',''), gh.get('_source',''), gt.get('_source','')]
        snapshot['_fetched_at'] = datetime.utcnow().isoformat()+'Z'
        return snapshot

# ---------------------- Enhanced Career Matcher ----------------------
class IntelligentCareerMatcher:
    def __init__(self, demand_snapshot: Dict[str, dict]):
        self.snapshot = demand_snapshot

    def fresh_graduate_scoring(self, features: Dict) -> Dict[str, float]:
        """Special scoring logic for fresh graduates"""
        career_scores = features.get('career_scores', {})
        skills = features.get('skills', [])
        coursework = features.get('coursework', [])
       
        # Boost scores based on coursework relevance
        coursework_text = ' '.join(coursework).lower()
       
        adjusted_scores = {}
        for career in CAREER_PATHS.keys():
            base_score = career_scores.get(career, 0)
           
            # Coursework alignment bonus
            coursework_bonus = 0
            career_skills = SKILL_KEYWORDS.get(career, {})
            all_career_skills = []
            for cat_skills in career_skills.values():
                all_career_skills.extend(cat_skills)
           
            for skill in all_career_skills:
                if skill in coursework_text:
                    coursework_bonus += 3
           
            # Entry-level keyword bonus
            entry_keywords = CAREER_PATHS[career].get('entry_keywords', [])
            entry_bonus = 0
            for keyword in entry_keywords:
                if any(keyword in skill.lower() for skill in skills):
                    entry_bonus += 5
           
            adjusted_scores[career] = base_score + coursework_bonus + entry_bonus
       
        return adjusted_scores

    def adjust_by_market(self, base_probs: torch.Tensor, features: Dict) -> torch.Tensor:
        adjusted = base_probs.clone()
        careers = list(CAREER_PATHS.keys())
       
        # Special handling for fresh graduates
        if features.get('is_fresh_graduate', False):
            fresh_scores = self.fresh_graduate_scoring(features)
            max_fresh_score = max(fresh_scores.values()) if fresh_scores.values() else 1
           
            for i, career in enumerate(careers):
                # Fresh graduate boost based on skills and coursework
                fresh_boost = fresh_scores.get(career, 0) / max_fresh_score if max_fresh_score > 0 else 0
                market_demand = self.snapshot.get(career, {}).get('demand_index', 0.5)
               
                # Combined boost for fresh graduates
                total_boost = 1.0 + 0.3 * fresh_boost + 0.2 * market_demand
                adjusted[0][i] *= total_boost
        else:
            # Regular market adjustment for experienced professionals
            for i, career in enumerate(careers):
                di = self.snapshot.get(career, {}).get('demand_index', 0.5)
                exp = features.get('years_experience', 0)
                edu = features.get('education_level', 0)
                boost = 1.0 + 0.4*di + 0.05*min(exp,10) + (0.08 if edu>=4 else 0.0)
                adjusted[0][i] *= boost
       
        return F.softmax(adjusted, dim=1)

    def insight_stub(self, career: str, features: Dict) -> Dict:
        # Map demand index to qualitative growth
        di = self.snapshot.get(career, {}).get('demand_index', 0.5)
        growth = "High" if di >= 0.66 else ("Medium" if di >= 0.4 else "Stable")
       
        recs = []
        is_fresh_grad = features.get('is_fresh_graduate', False)
        exp_years = features.get('years_experience', 0)
       
        # Career-specific advice based on the matched career
        if career == 'Software Engineer':
            if is_fresh_grad or exp_years < 2:
                recs += [
                    "Master full-stack development with a focus on popular frameworks",
                    "Build a strong GitHub portfolio with 3-5 diverse projects",
                    "Practice data structures and algorithms for technical interviews",
                    "Contribute to open-source projects to gain visibility"
                ]
            else:
                recs += [
                    "Specialize in system design and architecture patterns",
                    "Lead technical initiatives and mentor junior developers",
                    "Stay updated with cloud technologies and DevOps practices"
                ]
        
        elif career == 'Data Scientist':
            if is_fresh_grad or exp_years < 2:
                recs += [
                    "Build end-to-end machine learning projects with real datasets",
                    "Master Python data science stack (pandas, scikit-learn, matplotlib)",
                    "Learn cloud platforms (AWS, GCP, or Azure) for ML deployment",
                    "Participate in Kaggle competitions to showcase your skills"
                ]
            else:
                recs += [
                    "Focus on MLOps and production machine learning systems",
                    "Develop expertise in deep learning and advanced algorithms",
                    "Lead data science initiatives and business impact projects"
                ]
        
        elif career == 'DevOps Engineer':
            if is_fresh_grad or exp_years < 2:
                recs += [
                    "Get hands-on with containerization (Docker) and orchestration (Kubernetes)",
                    "Learn Infrastructure as Code tools like Terraform or CloudFormation",
                    "Practice CI/CD pipeline setup with Jenkins, GitLab CI, or GitHub Actions",
                    "Understand monitoring and logging tools (Prometheus, Grafana, ELK stack)"
                ]
            else:
                recs += [
                    "Architect scalable infrastructure solutions",
                    "Implement advanced monitoring and observability practices",
                    "Lead cloud migration and optimization projects"
                ]
        
        elif career == 'Cybersecurity Specialist':
            if is_fresh_grad or exp_years < 2:
                recs += [
                    "Obtain foundational certifications (CompTIA Security+, CEH)",
                    "Practice on cybersecurity platforms like TryHackMe or Hack The Box",
                    "Learn network security fundamentals and common attack vectors",
                    "Develop skills in security tools and incident response procedures"
                ]
            else:
                recs += [
                    "Pursue advanced certifications (CISSP, CISM, OSCP)",
                    "Specialize in areas like cloud security or threat intelligence",
                    "Lead security assessments and develop security policies"
                ]
        
        elif career == 'Cloud Engineer':
            if is_fresh_grad or exp_years < 2:
                recs += [
                    "Get certified in at least one major cloud platform (AWS, Azure, or GCP)",
                    "Learn Infrastructure as Code and cloud architecture patterns",
                    "Practice with serverless technologies and microservices",
                    "Understand cloud security and cost optimization principles"
                ]
            else:
                recs += [
                    "Design enterprise-scale cloud solutions",
                    "Implement multi-cloud and hybrid cloud strategies",
                    "Lead cloud transformation initiatives"
                ]
        
        elif career == 'Product Manager':
            if is_fresh_grad or exp_years < 2:
                recs += [
                    "Learn product management frameworks (Jobs-to-be-Done, OKRs)",
                    "Develop user research and data analysis skills",
                    "Build a portfolio of product case studies or mock products",
                    "Understand agile development and work closely with engineering teams"
                ]
            else:
                recs += [
                    "Focus on strategic product vision and roadmap planning",
                    "Develop advanced analytics and A/B testing expertise",
                    "Lead cross-functional teams and stakeholder management"
                ]
        
        elif career == 'UI/UX Designer':
            if is_fresh_grad or exp_years < 2:
                recs += [
                    "Master design tools (Figma, Sketch, Adobe Creative Suite)",
                    "Build a strong portfolio showcasing user-centered design process",
                    "Learn user research methods and usability testing",
                    "Understand front-end development basics to collaborate effectively"
                ]
            else:
                recs += [
                    "Develop expertise in design systems and accessibility",
                    "Lead design strategy and mentor junior designers",
                    "Specialize in emerging areas like voice UI or AR/VR design"
                ]
        
        elif career == 'Business Analyst':
            if is_fresh_grad or exp_years < 2:
                recs += [
                    "Learn business process modeling and requirements gathering techniques",
                    "Develop proficiency in data analysis tools (SQL, Excel, Tableau)",
                    "Understand project management methodologies (Agile, Waterfall)",
                    "Practice stakeholder communication and documentation skills"
                ]
            else:
                recs += [
                    "Focus on digital transformation and process improvement initiatives",
                    "Develop advanced analytics and business intelligence skills",
                    "Lead requirement gathering for complex enterprise projects"
                ]
        
        elif career == 'Marketing Specialist':
            if is_fresh_grad or exp_years < 2:
                recs += [
                    "Master digital marketing channels (SEO, SEM, social media, email)",
                    "Learn marketing analytics tools (Google Analytics, Facebook Ads Manager)",
                    "Develop content creation and copywriting skills",
                    "Understand customer journey mapping and conversion optimization"
                ]
            else:
                recs += [
                    "Focus on growth marketing and advanced attribution modeling",
                    "Lead integrated marketing campaigns and brand strategy",
                    "Develop expertise in marketing automation and CRM systems"
                ]
        
        elif career == 'Sales Representative':
            if is_fresh_grad or exp_years < 2:
                recs += [
                    "Learn consultative selling and relationship building techniques",
                    "Master CRM systems and sales pipeline management",
                    "Develop strong communication and presentation skills",
                    "Understand your target market and customer pain points deeply"
                ]
            else:
                recs += [
                    "Focus on strategic account management and territory expansion",
                    "Develop expertise in sales forecasting and analytics",
                    "Lead sales team initiatives and mentor junior sales staff"
                ]
        
        # General recommendations
        if is_fresh_grad or exp_years < 2:
            recs += [
                "Build a professional network through LinkedIn and industry events",
                "Seek mentorship from experienced professionals in your field"
            ]
        
        return {
            'growth_outlook': growth,
            'recommendations': recs,
            'is_fresh_graduate_advice': is_fresh_grad
        }


# ---------------------- Enhanced PyTorch Model for Fresh Graduates ----------------------
class FreshGradCareerPredictor(nn.Module):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', num_careers=10):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.num_careers = num_careers

        # Don't freeze any layers for better fresh graduate learning
        for param in self.encoder.parameters():
            param.requires_grad = True

        # Enhanced architecture for fresh graduates
        self.text_classifier = nn.Sequential(
            nn.Linear(384, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
       
        # Feature processing branch for skills, education, etc.
        self.feature_processor = nn.Sequential(
            nn.Linear(50, 128),  # Process extracted features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
       
        # Combined classifier
        self.final_classifier = nn.Sequential(
            nn.Linear(256 + 64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_careers)
        )
       
        self.temperature = nn.Parameter(torch.ones(1) * 0.8)  # Lower temperature for sharper predictions
        self.eval()

    def encode_text(self, text: str):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            out = self.encoder(**inputs)
            # Enhanced text encoding with attention to key sections
            embeddings = out.last_hidden_state
           
            # Weighted average based on attention scores
            attention_weights = torch.softmax(torch.sum(embeddings ** 2, dim=-1), dim=-1)
            weighted_embedding = torch.sum(embeddings * attention_weights.unsqueeze(-1), dim=1)
           
            return weighted_embedding

    def process_features(self, features: Dict):
        """Convert extracted features to tensor"""
        feature_vec = torch.zeros(50)
       
        # Years of experience (normalized)
        exp_years = min(features.get('years_experience', 0), 10) / 10.0
        feature_vec[0] = exp_years
       
        # Education level (one-hot encoded)
        edu_level = features.get('education_level', 0)
        if 0 <= edu_level <= 5:
            feature_vec[edu_level + 1] = 1.0
       
        # Skills count per career (top 10 careers)
        career_scores = features.get('career_scores', {})
        max_score = max(career_scores.values()) if career_scores.values() else 1
       
        careers = list(CAREER_PATHS.keys())
        for i, career in enumerate(careers):
            if i < 10:  # Limit to 10 careers
                normalized_score = career_scores.get(career, 0) / max_score if max_score > 0 else 0
                feature_vec[7 + i] = normalized_score
       
        # Seniority indicators
        seniority = features.get('seniority_indicators', {})
        seniority_keys = ['leadership', 'management', 'mentoring', 'technical', 'achievement']
        for i, key in enumerate(seniority_keys):
            if i < 10:  # Space for seniority features
                feature_vec[17 + i] = min(seniority.get(key, 0), 10) / 10.0
       
        # Fresh graduate indicator
        feature_vec[27] = 1.0 if features.get('is_fresh_graduate', False) else 0.0
       
        # Skills diversity (number of unique skills normalized)
        skills_count = min(len(features.get('skills', [])), 50) / 50.0
        feature_vec[28] = skills_count
       
        # Coursework relevance
        coursework_count = min(len(features.get('coursework', [])), 20) / 20.0
        feature_vec[29] = coursework_count
       
        return feature_vec

    def forward(self, text: str, features: Dict = None):
        self.eval()
        with torch.no_grad():
            # Text encoding
            text_emb = self.encode_text(text)
            text_features = self.text_classifier(text_emb)
           
            if features:
                # Feature processing
                feature_tensor = self.process_features(features).unsqueeze(0)
                feature_features = self.feature_processor(feature_tensor)
               
                # Combine text and features
                combined = torch.cat([text_features, feature_features], dim=1)
            else:
                # Use text features only with zero padding
                zero_features = torch.zeros(1, 64)
                combined = torch.cat([text_features, zero_features], dim=1)
           
            # Final prediction
            logits = self.final_classifier(combined)
            scaled = logits / self.temperature
            return F.softmax(scaled, dim=-1)

# ---------------------- Local DeepSeek reranker ----------------------
try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

class LLMReranker:
    def __init__(self):
        self.local_model = "deepseek-r1:1.5b"

    def rerank(self, resume_text: str, top_candidates: list,
               snapshot: dict, features: dict, location_hint: str = "India") -> dict:

        if not HAS_OLLAMA:
            return self._fallback_ranking(top_candidates, features)

        try:
            # Create better context for fresh graduates
            is_fresh_grad = features.get('is_fresh_graduate', False)
            exp_years = features.get('years_experience', 0)
            skills = ', '.join(features.get('skills', [])[:10])
            coursework = ', '.join(features.get('coursework', [])[:5])
           
            context = f"""
Profile Analysis:
- Experience: {exp_years} years ({'Fresh Graduate' if is_fresh_grad else 'Experienced'})
- Key Skills: {skills}
- Education: {features.get('education_level', 0)}/5 level
- Coursework: {coursework}

Top AI Predictions: {[(c, f"{p*100:.1f}%") for c, p in top_candidates[:5]]}
"""

            prompt = f"""You are a career counselor specializing in fresh graduates and entry-level positions.

{context}

RESUME EXCERPT:
{resume_text[:500]}

TASK: Analyze and rank top 3 career matches. Consider the candidate's current skill level and provide realistic entry-level advice.

Return ONLY this JSON structure (no additional text):
{{
  "paths": [
    {{
      "career": "exact_career_name_from_predictions",
      "reason": "why this career fits based on skills and education",
      "salary_range": "entry-level salary for {location_hint}",
      "missing_skills": ["specific skills to learn"],
      "confidence": 0.75,
      "entry_role": "appropriate entry-level position"
    }}
  ]
}}"""

            response = ollama.chat(
                model=self.local_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2, "num_predict": 500}
            )

            txt = response['message']['content'].strip()
           
            # Extract JSON from response
            json_start = txt.find('{')
            json_end = txt.rfind('}') + 1
           
            if json_start >= 0 and json_end > json_start:
                json_str = txt[json_start:json_end]
                data = json.loads(json_str)
               
                if "paths" in data and len(data["paths"]) > 0:
                    return {"engine": "ollama", "model": self.local_model, "data": data}

        except json.JSONDecodeError as e:
            st.warning(f"DeepSeek JSON parsing error: {e}")
        except Exception as e:
            st.warning(f"DeepSeek error: {e}")

        return self._fallback_ranking(top_candidates, features)

    def _fallback_ranking(self, top_candidates, features):
        is_fresh_grad = features.get('is_fresh_graduate', False)
        resume_skills = set([s.lower() for s in features.get("skills", [])])

        fallback = []
        for career, prob in top_candidates[:5]:  # check top 5 instead of just 3
            core_skills = set(SKILL_KEYWORDS.get(career, {}).get("core", []))
            overlap = len(resume_skills & core_skills)

        # Filter out irrelevant careers (no core skill overlap)
            if overlap == 0 and prob < 0.2:  
                continue  # skip this career

            entry_roles = CAREER_PATHS.get(career, {}).get('fresh_grad_roles', ['Junior ' + career])
            fallback.append({
            "career": career,
            "reason": f"Strong skill alignment detected. {'Ideal for fresh graduates.' if is_fresh_grad else 'Good experience match.'}",
            "salary_range": "3-6 LPA" if is_fresh_grad else "6-12 LPA",
            "missing_skills": [],
            "confidence": round(float(prob), 3),
            "entry_role": entry_roles[0] if entry_roles else career
        })

        # Stop after 3 valid matches
            if len(fallback) >= 3:
                break

        return {"engine": "fallback", "model": "rule-based", "data": {"paths": fallback}}

# ---------------------- Enhanced Pipeline for Fresh Graduates ----------------------
@st.cache_resource
def load_fresh_grad_model():
    """Cache the enhanced model for fresh graduates"""
    return FreshGradCareerPredictor()

def predict_pipeline(text: str, model: FreshGradCareerPredictor, location_hint: str="India",
                     github_token: str | None = None) -> Dict:
    processor = AdvancedResumeProcessor()
    features = processor.extract_features(text)

    # Enhanced model predictions with features
    base = model(text, features)
    careers = list(CAREER_PATHS.keys())
   
    # Get top predictions
    top_values, top_indices = torch.topk(base, min(len(careers), 5), dim=1)
    top_prob = top_values[0].tolist()
    top_idx = top_indices[0].tolist()

    # Real-time market signals
    rts = RealTimeSignals(github_token=github_token, location_hint=location_hint)
    snapshot = rts.career_demand_snapshot(features)

    # Market-aware adjustment with fresh graduate considerations
    matcher = IntelligentCareerMatcher(snapshot)
    adjusted = matcher.adjust_by_market(base, features)

    # Get adjusted rankings
    adj_values, adj_indices = torch.topk(adjusted, min(len(careers), 5), dim=1)
    top_prob_adj = adj_values[0].tolist()
    top_idx_adj = adj_indices[0].tolist()
   
    ranked = [(careers[top_idx_adj[i]], float(top_prob_adj[i])) for i in range(len(top_idx_adj))]

    # Enhanced ranking for fresh graduates using rule-based scoring
    if features.get('is_fresh_graduate', False):
        fresh_scores = matcher.fresh_graduate_scoring(features)
        max_fresh_score = max(fresh_scores.values()) if fresh_scores.values() else 1
       
        # Combine AI predictions with fresh graduate scoring
        combined_scores = {}
        for career, ai_score in ranked:
            fresh_score = fresh_scores.get(career, 0) / max_fresh_score if max_fresh_score > 0 else 0
            # 50% AI, 50% rule-based for fresh graduates
            combined_score = 0.5 * ai_score + 0.5 * fresh_score
            combined_scores[career] = combined_score
       
        # Re-rank based on combined scores
        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

    primary_career = ranked[0][0]
    insights = matcher.insight_stub(primary_career, features)

    # Enhanced LLM reranker with fresh graduate context
    reranker = LLMReranker()
    llm = reranker.rerank(text, ranked[:5], snapshot, features, location_hint=location_hint)

    # Merge LLM outputs
    llm_paths = llm["data"].get("paths", [])
    salary_map = {p["career"]: p.get("salary_range","N/A") for p in llm_paths if "career" in p}
    conf_map = {p["career"]: float(p.get("confidence",0)) for p in llm_paths if "career" in p}
    entry_role_map = {p["career"]: p.get("entry_role", p["career"]) for p in llm_paths if "career" in p}

    missing_skills_map = {p["career"]: p.get("missing_skills", []) for p in llm_paths if "career" in p}


    # Career progression timeline (enhanced for fresh graduates)
    exp_years = features.get('years_experience', 0)
    edu_level = features.get('education_level', 0)
    is_fresh_grad = features.get('is_fresh_graduate', False)
   
    if is_fresh_grad:
        # Fresh graduate progression
        base_years = 2 if edu_level >= 4 else 3  # Master's vs Bachelor's
        progression = [
            (entry_role_map.get(primary_career, f"Junior {primary_career}"), 0),  # Start immediately
        ]
        for i, role in enumerate(CAREER_PATHS[primary_career]['progression'][:3]):
            years_to_role = base_years + i * 2
            progression.append((role, years_to_role))
    else:
        # Regular progression
        seniority_total = sum(features.get('seniority_indicators', {}).values())
        base_years = max(1, 3 - exp_years//2)
        if edu_level >= 4:
            base_years -= 0.5
        if seniority_total > 5:
            base_years -= 1
       
        progression = []
        roles = CAREER_PATHS[primary_career]['progression'][:4]
        for i, role in enumerate(roles):
            years_to_role = max(1, int(base_years + i * 1.8))
            progression.append((role, years_to_role))

    return {
        "primary_career": primary_career,
        "probability": float(ranked[0][1]),
        "alternatives": ranked[1:4] if len(ranked) > 1 else [],
        "progression": progression,
        "features": features,
        "insights": insights,
        "market_snapshot": snapshot,
        "llm": llm,
        "salary_estimates": salary_map,
        "llm_confidence": conf_map,
        "entry_roles": entry_role_map,
        "missing_skills": missing_skills_map,
        "career_scores_raw": features.get('career_scores', {}),
        "is_fresh_graduate": features.get('is_fresh_graduate', False)
    }

# ---------------------- Professional Streamlit UI ----------------------
def main():
    st.set_page_config(page_title="AI Career Intelligence Platform", page_icon="🎯", layout="wide")
    
    # Professional header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; font-size: 2.5rem; margin: 0;">🎯 AI Career Intelligence Platform</h1>
        <p style="color: #f0f0f0; font-size: 1.2rem; margin-top: 0.5rem;">Advanced AI-powered career matching and guidance system</p>
    </div>
    """, unsafe_allow_html=True)

    # File upload section
    uploaded_file = st.file_uploader(
        "📄 Upload Your Resume",
        type=['pdf', 'docx'],
        help="Upload a PDF or Word document containing your resume for AI analysis"
    )

    if uploaded_file is not None:
        processor = AdvancedResumeProcessor()
       
        # Extract text based on file type
        if uploaded_file.type == "application/pdf":
            resume_text = processor.extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
            resume_text = processor.extract_text_from_docx(uploaded_file)
        else:
            st.error("❌ Unsupported file format")
            return

        if not resume_text:
            st.error("❌ Could not extract text from the uploaded file. Please try a different file.")
            return

        # Check if it's actually a resume
        if not processor.is_resume(resume_text):
            st.warning("⚠️ This doesn't appear to be a resume. Please upload a proper resume document.")
            return

        # Clean the text
        clean_text = processor.clean(resume_text)
       
        with st.spinner("🤖 Analyzing your career profile..."):
            try:
                model = load_fresh_grad_model()
                results = predict_pipeline(
                    clean_text,
                    model,
                    location_hint="India",
                    github_token=None
                )
               
                # Display results with professional styling
                col1, col2 = st.columns([2, 1])
               
                with col1:
                    # Primary career prediction
                    primary_career = results['primary_career']
                    confidence = results['probability'] * 100
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                color: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem; 
                                box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
                        <h2 style="margin: 0 0 0.5rem 0; font-size: 2rem;">🎯 Best Career Match</h2>
                        <h3 style="margin: 0; font-size: 1.8rem; color: #fff;">{primary_career}</h3>
                        <div style="background: rgba(255,255,255,0.2); padding: 0.5rem; border-radius: 5px; margin-top: 1rem;">
                            <strong>Confidence Score: {confidence:.1f}%</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                   
                    # Entry-level role suggestion
                    entry_roles = results.get('entry_roles', {})
                    if primary_career in entry_roles:
                        st.success(f"🚀 **Recommended Starting Position:** {entry_roles[primary_career]}")
                   
                    # Market insights
                    insights = results['insights']
                    
                    col1_1, col1_2 = st.columns(2)
                    with col1_1:
                        growth_color = "🟢" if insights['growth_outlook'] == "High" else "🟡" if insights['growth_outlook'] == "Medium" else "🔵"
                        st.info(f"📈 **Market Growth:** {growth_color} {insights['growth_outlook']}")
                    
                    with col1_2:
                        missing_skills = results.get("missing_skills", {}).get(primary_career, [])
                        skills_count = len(missing_skills)
                        if skills_count > 0:
                            st.warning(f"🛠️ **Skills to Develop:** {skills_count} identified")
                   
                    # Alternative careers
                    st.subheader("🔄 Alternative Career Paths")
                    alternatives = results.get('alternatives', [])
                    if alternatives:
                        for i, (career, prob) in enumerate(alternatives, 2):
                            entry_role = entry_roles.get(career, f"Junior {career}")
                            match_strength = "Strong" if prob > 0.7 else "Good" if prob > 0.5 else "Fair"
                            st.write(f"**{i}.** **{career}** • {prob*100:.1f}% match ({match_strength}) → *{entry_role}*")
                   
                    # Career progression timeline
                    st.subheader("📈 Career Progression Timeline")
                    
                    progression_html = "<div style='background: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;'>"
                    for i, (role, years) in enumerate(results['progression']):
                        if years == 0:
                            progression_html += f"<div style='margin: 0.5rem 0; padding: 0.5rem; background: #e3f2fd; border-left: 4px solid #2196f3; border-radius: 5px;'><strong>🎯 {role}</strong> — <em>Start here</em></div>"
                        else:
                            progression_html += f"<div style='margin: 0.5rem 0; padding: 0.5rem; background: #f3e5f5; border-left: 4px solid #9c27b0; border-radius: 5px;'><strong>📊 {role}</strong> — <em>in {years} years</em></div>"
                    progression_html += "</div>"
                    
                    st.markdown(progression_html, unsafe_allow_html=True)
               
                with col2:
                    st.markdown("""
                    <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
                        <h3 style="color: #333; margin-top: 0;">📊 Profile Analysis</h3>
                    </div>
                    """, unsafe_allow_html=True)
                   
                    features = results['features']
                   
                    # Experience and Education
                    exp_display = f"{features['years_experience']:.1f} years"
                    if results.get('is_fresh_graduate', False):
                        exp_display += " (Entry-level)"
                   
                    col2_1, col2_2 = st.columns(2)
                    with col2_1:
                        st.metric("Experience", exp_display)
                    with col2_2:
                        education_levels = {0: "None", 1: "Certificate", 2: "Diploma/12th", 3: "Bachelor's", 4: "Master's", 5: "PhD"}
                        st.metric("Education", education_levels.get(features['education_level'], "Unknown"))
                   
                    # Skills found with better organization
                    if features['skills']:
                        st.subheader("🛠️ Skills Detected")
                        skills = features['skills'][:15]
                        
                        # Display skills in a more compact format
                        skills_display = ""
                        for i, skill in enumerate(skills):
                            if i > 0 and i % 3 == 0:
                                skills_display += "\n"
                            skills_display += f"• {skill}  "
                        
                        st.text_area("", skills_display, height=100, disabled=True)
                   
                    # Coursework for students
                    coursework = features.get('coursework', [])
                    if coursework:
                        st.subheader("📚 Relevant Coursework")
                        coursework_display = coursework[:8]
                        for course in coursework_display:
                            st.write(f"• {course}")
                   
                    # Experience indicators
                    seniority = features['seniority_indicators']
                    if any(seniority.values()):
                        st.subheader("🏆 Experience Indicators")
                        indicators_data = []
                        for category, count in seniority.items():
                            if count > 0:
                                indicators_data.append({"Category": category.title(), "Count": count})
                        
                        if indicators_data:
                            import pandas as pd
                            df = pd.DataFrame(indicators_data)
                            st.dataframe(df, hide_index=True, use_container_width=True)

                # Career-Specific Recommendations Section
                st.markdown("""
                <div style="background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); 
                            color: white; padding: 1.5rem; border-radius: 15px; margin: 2rem 0;">
                    <h2 style="margin: 0; color: white;">💡 Personalized Career Guidance</h2>
                </div>
                """, unsafe_allow_html=True)
                
                recs = insights.get('recommendations', [])
                
                if recs:
                    # Display recommendations in a clean format
                    for i, rec in enumerate(recs, 1):
                        st.markdown(f"""
                        <div style="background: #f8f9fa; padding: 1rem; margin: 0.5rem 0; 
                                    border-left: 4px solid #4CAF50; border-radius: 5px;">
                            <strong>{i}.</strong> {rec}
                        </div>
                        """, unsafe_allow_html=True)
                
                # LLM Analysis Section
                st.markdown("""
                <div style="background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%); 
                            color: white; padding: 1.5rem; border-radius: 15px; margin: 2rem 0;">
                    <h2 style="margin: 0; color: white;">🧠 AI Career Counselor Analysis</h2>
                </div>
                """, unsafe_allow_html=True)
                
                llm_data = results['llm']['data']
               
                if llm_data.get('paths'):
                    for i, path in enumerate(llm_data['paths'][:3], 1):
                        confidence = path.get('confidence', 0)
                        conf_color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.5 else "🔴"
                        
                        with st.expander(f"{conf_color} #{i} {path['career']} - {confidence*100:.0f}% confidence"):
                            st.write(f"**Analysis:** {path.get('reason', 'No explanation provided')}")
                           
                            col_info1, col_info2 = st.columns(2)
                            
                            with col_info1:
                                if path.get('entry_role'):
                                    st.info(f"**Starting Role:** {path['entry_role']}")
                                if path.get('salary_range') and path['salary_range'] != 'N/A':
                                    st.success(f"**Salary Range:** {path['salary_range']}")
                            
                            with col_info2:
                                if path.get('missing_skills'):
                                    skills_to_learn = path['missing_skills'][:5]
                                    st.warning(f"**Skills to Develop:** {', '.join(skills_to_learn)}")
                else:
                    st.info("AI counselor analysis not available. Using core predictions only.")

            except Exception as e:
                st.error(f"Error analyzing resume: {str(e)}")

    else:
        # Professional landing page without hardcoded values
        st.markdown("""
        <div style="text-align: center; padding: 3rem 0;">
            <div style="background: white; padding: 3rem; border-radius: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); margin: 2rem 0;">
                <h2 style="color: #333; margin-bottom: 2rem;">🚀 Advanced AI Career Matching</h2>
                <p style="font-size: 1.2rem; color: #666; line-height: 1.8; max-width: 800px; margin: 0 auto;">
                    Upload your resume to receive AI-powered career predictions, personalized guidance, 
                    and actionable insights tailored to your unique professional profile.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Feature highlights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background: #e3f2fd; padding: 2rem; border-radius: 15px; text-align: center; height: 250px;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🎯</div>
                <h3 style="color: #1976d2; margin-bottom: 1rem;">Precise Career Matching</h3>
                <p style="color: #555; line-height: 1.6;">
                    Advanced machine learning algorithms analyze your skills, experience, 
                    and background to identify optimal career paths.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: #f3e5f5; padding: 2rem; border-radius: 15px; text-align: center; height: 250px;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📈</div>
                <h3 style="color: #7b1fa2; margin-bottom: 1rem;">Market Intelligence</h3>
                <p style="color: #555; line-height: 1.6;">
                    Real-time market demand analysis from multiple data sources 
                    to guide your career decisions with current industry insights.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: #e8f5e8; padding: 2rem; border-radius: 15px; text-align: center; height: 250px;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">💡</div>
                <h3 style="color: #388e3c; margin-bottom: 1rem;">Personalized Guidance</h3>
                <p style="color: #555; line-height: 1.6;">
                    AI-powered career counselor provides customized recommendations, 
                    skill development paths, and career progression strategies.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # How it works section
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 3rem; border-radius: 20px; margin: 3rem 0;">
            <h2 style="text-align: center; margin-bottom: 2rem; color: white;">How It Works</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 2rem;">
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 1rem;">📄</div>
                    <h4 style="margin-bottom: 1rem; color: white;">1. Upload Resume</h4>
                    <p style="color: #f0f0f0;">Upload your PDF or Word resume for comprehensive analysis</p>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 1rem;">🤖</div>
                    <h4 style="margin-bottom: 1rem; color: white;">2. AI Analysis</h4>
                    <p style="color: #f0f0f0;">Advanced algorithms extract and analyze your professional profile</p>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 1rem;">🎯</div>
                    <h4 style="margin-bottom: 1rem; color: white;">3. Career Matching</h4>
                    <p style="color: #f0f0f0;">Get ranked career recommendations with confidence scores</p>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 1rem;">💡</div>
                    <h4 style="margin-bottom: 1rem; color: white;">4. Actionable Insights</h4>
                    <p style="color: #f0f0f0;">Receive personalized guidance and development roadmaps</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Supported career paths
        st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin: 2rem 0;">
            <h3 style="text-align: center; color: #333; margin-bottom: 2rem;">🎯 Career Paths We Support</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Display career paths in a grid
        career_names = list(CAREER_PATHS.keys())
        career_cols = st.columns(2)
        
        for i, career in enumerate(career_names):
            with career_cols[i % 2]:
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 1rem; margin: 0.5rem 0; 
                            border-left: 4px solid #667eea; border-radius: 5px;">
                    <strong>{career}</strong>
                </div>
                """, unsafe_allow_html=True)

        # Privacy and security note
        st.markdown("""
        <div style="background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; 
                    padding: 1rem; border-radius: 10px; margin: 2rem 0; text-align: center;">
            <strong>🔐 Privacy Assured:</strong> Your resume data is processed securely and not stored permanently. 
            All analysis is performed in real-time for your privacy and security.
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()