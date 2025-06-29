import os
import shutil
import json
from datetime import datetime
from janome.tokenizer import Tokenizer
from collections import Counter, defaultdict
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import MeCab
from textstat import flesch_reading_ease
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
rcParams['font.family'] = 'IPAGothic'
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")
sns.set_palette("husl")

class AdvancedTextMiner:
    """高度なテキストマイニング分析システム（推論・感情語・構造語除去機能強化版）"""
    
    def __init__(self, config_path=None):
        # 設定の初期化
        self.config = self._load_config(config_path) if config_path else self._default_config()
        self.tokenizer = Tokenizer()
        
        # 改良版：言語学的カテゴリ別除外語辞書の初期化
        self._init_linguistic_filters()
        
        # MeCabの設定と詳細な状態チェック
        self.use_mecab = False
        self.mecab = None
        self._setup_mecab()
            
        self.results = {}
    
    def _init_linguistic_filters(self):
        """言語学的カテゴリ別の除外語辞書を初期化"""
        
        # 1. 推論・感情語カテゴリ（認知・感情・推測を表す語彙）
        self.inference_emotion_words = {
            # 推論・推測動詞
            'おもう', '思う', 'かんがえる', '考える', 'しんじる', '信じる', 
            'すいそく', '推測', 'すいろん', '推論', 'すいてい', '推定',
            'よそう', '予想', 'きたい', '期待', 'りかい', '理解',
            
            # 可能性・推量表現（様態表現を強化）
            'しれる', 'かもしれない', 'だろう', 'であろう', 'かもしれません',
            'らしい', 'ようだ', 'ような', 'ように', 'よう', 'みたい', 'みたいな',
            'っぽい', 'げ', 'がち', 'そう', 'そうだ', 'そうな',
            
            # 感情動詞
            'かんじる', '感じる', 'よろこぶ', '喜ぶ', 'かなしむ', '悲しむ',
            'おこる', '怒る', 'おどろく', '驚く', 'しんぱい', '心配',
            'あんしん', '安心', 'こうかい', '後悔', 'まんぞく', '満足',
            
            # 感情形容詞
            'うれしい', '嬉しい', 'かなしい', '悲しい', 'たのしい', '楽しい',
            'つらい', 'しんどい', 'きもちい', '気持ちい', 'いやだ', '嫌だ',
            
            # 評価・判断語
            'ひょうか', '評価', 'はんだん', '判断', 'いけん', '意見',
            'かんそう', '感想', 'いんしょう', '印象', 'かんてん', '観点'
        }
        
        # 2. 構造語カテゴリ（文章構造や論理関係を示す語彙）
        self.structural_words = {
            # 接続詞・接続副詞
            'しかし', 'だが', 'けれど', 'けれども', 'ところが', 'でも',
            'そして', 'それから', 'つぎに', '次に', 'さらに', 'また',
            'だから', 'それで', 'ゆえに', 'したがって', 'そのため',
            'つまり', 'すなわち', 'ようするに', '要するに', 'いいかえれば', '言い換えれば',
            'たとえば', '例えば', 'ちなみに', 'ところで', 'さて',
            'ただし', 'もっとも', 'なお', 'ちなみに', 'いっぽう', '一方',
            
            # 文章構造語
            'はじめに', '初めに', 'つぎに', '次に', 'さいごに', '最後に',
            'けっろん', '結論', 'ようやく', 'まとめ', 'せつめい', '説明',
            'りゆう', '理由', 'げんいん', '原因', 'けっか', '結果',
            'もくてき', '目的', 'ほうほう', '方法', 'しゅだん', '手段',
            
            # 程度・頻度副詞
            'とても', 'かなり', 'ずいぶん', 'だいぶ', 'わりと', 'けっこう',
            'すこし', '少し', 'ちょっと', 'やや', 'わずか', 'ほんの',
            'たいへん', '大変', 'ひじょう', '非常', 'きわめて', '極めて',
            'いつも', '常に', 'よく', 'ときどき', '時々', 'たまに',
            'ぜんぜん', '全然', 'まったく', '全く', 'けっして', '決して',
            
            # 時間・空間指示語
            'いま', '今', 'きょう', '今日', 'きのう', '昨日', 'あした', '明日',
            'ここ', 'そこ', 'あそこ', 'うえ', '上', 'した', '下', 'まえ', '前',
            'まわり', '周り', 'あいだ', '間', 'なか', '中', 'そと', '外',
            
            # 敬語・丁寧語（拡張）
            'いらっしゃる', 'くださる', 'れる', 'られる', 'せる', 'させる',
            'です', 'ます', 'であります', 'ございます', 'いたします'
        }
        
        # 3. 機能語カテゴリ（拡張版）
        self.functional_words = {
            # 基本的な機能語
            'する', 'ある', 'いる', 'なる', 'くださる', 'ください', 'である', 'だ', 'です', 'ます',
            
            # 形式名詞（拡張）
            'こと', 'もの', 'ところ', 'ため', 'わけ', 'はず', 'つもり', 'ほう', 'うち', 
            'とき', '時', 'ばあい', '場合', 'とおり', '通り', 'たび', '度',
            'さい', '際', 'あいだ', '間', 'まえ', '前', 'あと', '後',
            
            # 代名詞（拡張）
            'これ', 'それ', 'あれ', 'この', 'その', 'あの', 'ここ', 'そこ', 'あそこ',
            'どれ', 'どの', 'どこ', 'だれ', '誰', 'なに', '何', 'いつ', 'なぜ', 'どう',
            
            # 一般的すぎる動詞（拡張）
            'できる', 'みる', '見る', 'いく', '行く', 'くる', '来る', 'おく', '置く', 'やる',
            'もつ', '持つ', 'いう', '言う', 'きく', '聞く', 'かく', '書く', 'よむ', '読む',
            'たべる', '食べる', 'のむ', '飲む', 'ねる', '寝る', 'おきる', '起きる',
            'はたらく', '働く', 'べんきょう', '勉強', 'しごと', '仕事',
            
            # 助数詞的なもの（拡張）
            '回', '度', '件', '個', '人', '名', '本', '冊', '枚', '台', '機', '社', '校',
            '日', '月', '年', '時間', '分', '秒', 'つ', 'か月', 'ばん', '番'
        }
        
        # 4. 敬称・敬語カテゴリ（新設・強化）
        self.honorific_words = {
            # 敬称（人名に付く）
            'さん', 'ちゃん', 'くん', '様', 'さま', '氏', '君', 'さま', 'はん',
            
            # 敬語・丁寧語（拡張）
            'いらっしゃる', 'くださる', 'れる', 'られる', 'せる', 'させる',
            'です', 'ます', 'であります', 'ございます', 'いたします',
            'していただく', 'させていただく', 'おります', 'いらっしゃいます',
            
            # 謙譲語
            'いたす', 'させていただく', 'おうかがい', 'お伺い', 'おじゃま', 'お邪魔',
            'うかがう', '伺う', 'もうす', '申す', 'もうしあげる', '申し上げる'
        }
        
        # 5. 統合除外語セットの作成
        self.all_excluded_words = (
            self.inference_emotion_words | 
            self.structural_words | 
            self.functional_words |
            self.honorific_words  # 敬称カテゴリを追加
        )
        
        print(f"除外語辞書を初期化しました:")
        print(f"・推論・感情語: {len(self.inference_emotion_words)}語")
        print(f"・構造語: {len(self.structural_words)}語")
        print(f"・機能語: {len(self.functional_words)}語")
        print(f"・敬称・敬語: {len(self.honorific_words)}語")  # 新しいカテゴリ
        print(f"・総除外語数: {len(self.all_excluded_words)}語")
    
    def _setup_mecab(self):
        """MeCabの詳細な設定と状態チェック"""
        print("\n形態素解析エンジンの設定を確認中...")
        
        # Step 1: MeCabライブラリのインポート確認
        try:
            import MeCab
            print("✓ MeCabライブラリが見つかりました")
        except ImportError as e:
            print("❌ MeCabライブラリが見つかりません")
            print(f"   詳細: {e}")
            print("   → Janomeを使用します（十分な性能があります）")
            return
        
        # Step 2: MeCabタガーの初期化確認
        try:
            # 一般的な設定での初期化を試行
            self.mecab = MeCab.Tagger('-Ochasen')
            print("✓ MeCabタガーの初期化に成功しました")
        except Exception as e:
            print("❌ MeCabタガーの初期化に失敗しました")
            print(f"   詳細: {e}")
            
            # 辞書パスを明示的に指定して再試行
            try:
                self.mecab = MeCab.Tagger('-Ochasen -d /usr/lib/x86_64-linux-gnu/mecab/dic/ipadic-utf8')
                print("✓ 辞書パス指定でMeCabタガーの初期化に成功しました")
            except Exception as e2:
                print(f"❌ 辞書パス指定でも失敗: {e2}")
                print("   → Janomeを使用します")
                return
        
        # Step 3: 実際の動作テスト
        try:
            test_result = self.mecab.parse("テスト文章です。")
            if test_result and len(test_result.strip()) > 0:
                print("✓ MeCab動作テストに成功しました")
                print(f"   テスト結果: {test_result.strip()[:50]}...")
                self.use_mecab = True
                print("🎉 MeCabが正常に動作しています！高精度な解析を使用します")
            else:
                print("❌ MeCab動作テストで空の結果が返されました")
        except Exception as e:
            print(f"❌ MeCab動作テストに失敗: {e}")
            print("   → Janomeを使用します")
        
        if not self.use_mecab:
            print("\n📝 MeCabを使用したい場合は、以下のコマンドを試してください:")
            print("   sudo apt-get install python3-dev build-essential libmecab-dev")
            print("   pip install mecab-python3")
            print("\n💡 現在はJanomeを使用します。一般的な用途には十分な性能です！")
    
    def _default_config(self):
        """デフォルト設定（改良版フィルタリング機能付き）"""
        return {
            'source_dir': os.path.expanduser('~/Dropbox/text'),
            'archive_dir': os.path.expanduser('~/Dropbox/processed'),
            'output_dir': os.path.expanduser('~/Dropbox/results'),
            'from_email': "あなたのGmailアドレスを入力して下さい",
            'to_email': "送信先のメールアドレスを入力して下さい",
            'app_password': "Googleのアプリパスワードです",
            
            'min_word_length': 2,
            'min_frequency': 3,
            'network_top_n': 40,
            'topic_num': 5,
            'cluster_num': 7,
            
            # 新しい設定項目（フィルタリング強化）
            'enable_verb_normalization': True,      # 動詞の原形化を有効
            'strict_pos_filtering': True,           # 厳密な品詞フィルタリングを有効
            'exclude_inference_emotion': True,      # 推論・感情語の除外を有効
            'exclude_structural_words': True,       # 構造語の除外を有効
            'min_word_importance': 0.01,           # 語彙重要度の最小閾値
            'enable_semantic_filtering': True,      # 意味的フィルタリングを有効
        }
    
    def _load_config(self, path):
        """設定ファイルの読み込み"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def enhanced_tokenize(self, text):
        """言語学的知見に基づく高度な形態素解析（改良版フィルタリング）"""
        words = []
        
        if self.use_mecab:
            # MeCabを使用した高精度解析
            node = self.mecab.parseToNode(text)
            while node:
                if node.surface:
                    surface = node.surface
                    features = node.feature.split(',')
                    
                    if len(features) >= 4:
                        pos_major = features[0]
                        pos_minor1 = features[1]
                        pos_minor2 = features[2]
                        base_form = features[6] if len(features) > 6 and features[6] != '*' else surface
                        
                        # 改良版フィルタリング
                        if self._is_meaningful_word_enhanced(surface, base_form, pos_major, pos_minor1, pos_minor2):
                            # 動詞は原形を使用、その他は表層形を使用
                            word = base_form if pos_major == '動詞' and self.config['enable_verb_normalization'] else surface
                            words.append(word)
                
                node = node.next
        else:
            # Janomeによる解析（改良版フィルタリング）
            tokens = self.tokenizer.tokenize(text)
            for token in tokens:
                surface = token.surface
                pos_info = token.part_of_speech.split(',')
                
                if len(pos_info) >= 2:
                    pos_major = pos_info[0]
                    pos_minor1 = pos_info[1] if len(pos_info) > 1 else ''
                    pos_minor2 = pos_info[2] if len(pos_info) > 2 else ''
                    base_form = pos_info[6] if len(pos_info) > 6 and pos_info[6] != '*' else surface
                    
                    if self._is_meaningful_word_enhanced(surface, base_form, pos_major, pos_minor1, pos_minor2):
                        word = base_form if pos_major == '動詞' and self.config['enable_verb_normalization'] else surface
                        words.append(word)
        
        return words
    
    def _is_meaningful_word_enhanced(self, surface, base_form, pos_major, pos_minor1, pos_minor2):
        """改良版：語彙が分析対象として意味があるかを判定する高度フィルタ"""
        
        # 基本的な除外条件
        if (len(surface) < self.config['min_word_length'] or 
            surface.isascii() or 
            surface.isdigit()):
            return False
        
        # 改良版：カテゴリ別除外チェック（より確実な方法）
        if self.config.get('exclude_inference_emotion', True):
            if surface in self.inference_emotion_words or base_form in self.inference_emotion_words:
                return False
        
        if self.config.get('exclude_structural_words', True):
            if surface in self.structural_words or base_form in self.structural_words:
                return False
        
        # 機能語のチェック
        if surface in self.functional_words or base_form in self.functional_words:
            return False
            
        # 敬称・敬語のチェック（新設）
        if surface in self.honorific_words or base_form in self.honorific_words:
            return False
        
        # さらに詳細な除外パターン（敬称と様態表現を強化）
        additional_patterns = {
            # 敬称パターン（確実に除外）
            'honorifics_strict': {'さん', 'ちゃん', 'くん', '様', 'さま', '氏', '君'},
            
            # 様態表現（確実に除外）
            'modal_expressions': {
                'よう', 'ような', 'ように', 'ようだ', 'ようで', 'ようです',
                'みたい', 'みたいな', 'みたいに', 'みたいだ', 'みたいで',
                'っぽい', 'っぽく', 'っぽさ', 'らしい', 'らしく', 'らしさ'
            },
            
            # 人名パターン（拡張）
            'person_names': {'ジヒョ', 'チェヨン', 'ツウィ', 'ナヨン', 'モモ', 'サナ', 'ダヒョン', 'ジョンヨン', 'ミナ'},
            
            # 記号的表現
            'symbols': {'。', '、', '！', '？', ')', '(', '」', '「', '『', '』', '【', '】', '〈', '〉'},
            
            # 単位・助数詞
            'units': {'円', '万', '千', '百', '億', '兆', 'kg', 'km', 'cm', 'mm', 'g', 'ml', 'l'},
            
            # 一般的すぎる副詞
            'common_adverbs': {'とても', 'かなり', 'ずいぶん', 'だいぶ', 'わりと', 'けっこう', 'ちょっと', 'すこし', '少し'}
        }
        
        # パターンマッチング除外（確実性を向上）
        for pattern_type, pattern_set in additional_patterns.items():
            if surface in pattern_set or base_form in pattern_set:
                return False
        
        # 語尾による敬称チェック（さらなる確実性のため）
        honorific_suffixes = ['さん', 'ちゃん', 'くん', '様', 'さま', '氏']
        for suffix in honorific_suffixes:
            if surface.endswith(suffix) or base_form.endswith(suffix):
                return False
        
        # 様態表現の語尾チェック
        modal_suffixes = ['よう', 'ような', 'ように', 'みたい', 'らしい', 'っぽい']
        for suffix in modal_suffixes:
            if surface.endswith(suffix) or base_form.endswith(suffix):
                return False
        
        # 品詞による詳細フィルタリング（改良版）
        if pos_major == '名詞':
            if pos_minor1 in ['一般', '固有名詞', 'サ変接続']:
                # 固有名詞の場合、より厳密なチェック
                if pos_minor1 == '固有名詞':
                    # 人名らしきパターンを除外
                    if (len(surface) <= 4 and 
                        all(ord(char) >= 0x30A0 and ord(char) <= 0x30FF for char in surface)):
                        return False
                    # 組織名の一般的なパターンを除外
                    org_suffixes = {'会社', '株式会社', '有限会社', '財団法人', '社団法人', '大学', '学校', '病院'}
                    if any(surface.endswith(suffix) for suffix in org_suffixes):
                        return False
                return True
            elif pos_minor1 in ['代名詞', '数']:
                return False
            elif pos_minor2 in ['助数詞', '接尾', '非自立']:
                return False
            else:
                return len(surface) >= 2  # 短すぎる名詞を除外
                
        elif pos_major == '動詞':
            if pos_minor1 in ['自立']:
                # 一般的すぎる動詞の拡張リスト
                return base_form not in self.functional_words
            else:
                return False
                
        elif pos_major == '形容詞':
            if pos_minor1 in ['自立']:
                # 一般的すぎる形容詞を除外
                return base_form not in {'良い', 'よい', 'いい', '悪い', 'わるい', '多い', '少ない', '大きい', '小さい'}
            else:
                return False
        
        elif pos_major == '副詞':
            # 副詞は構造語に多く含まれるため、より厳しくフィルタリング
            if self.config.get('strict_pos_filtering', True):
                return False  # 副詞は基本的に除外
            else:
                return surface not in additional_patterns['common_adverbs']
        
        # その他の品詞は除外
        return False
    
    def extract_enhanced_features(self, text):
        """テキストから拡張された特徴量を抽出（改良版）"""
        words = self.enhanced_tokenize(text)
        
        # フィルタリング統計の出力
        if self.config.get('enable_semantic_filtering', True):
            print(f"フィルタリング後の語彙数: {len(words)}語")
        
        # 基本統計
        char_count = len(text)
        word_count = len(words)
        unique_words = len(set(words))
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        
        # 語彙の豊富さ（TTR: Type-Token Ratio）
        ttr = unique_words / word_count if word_count > 0 else 0
        
        # 共起ペアの抽出（改良版）
        weighted_pairs = Counter()
        
        # 動的ウィンドウサイズでの共起抽出
        window_sizes = [3, 5, 10]
        
        for window in window_sizes:
            weight = 1.0 / window
            for i in range(len(words)):
                for j in range(i + 1, min(i + window, len(words))):
                    pair = tuple(sorted((words[i], words[j])))
                    weighted_pairs[pair] += weight
        
        return {
            'words': words,
            'pairs': weighted_pairs,
            'word_count': word_count,
            'unique_words': unique_words,
            'char_count': char_count,
            'avg_word_length': avg_word_length,
            'ttr': ttr,
            'word_frequency': Counter(words)
        }
    
    def create_filtering_report(self):
        """フィルタリング効果のレポートを生成"""
        report = f"""
【フィルタリング設定レポート】

■ 除外語カテゴリ設定
・推論・感情語除外: {'有効' if self.config.get('exclude_inference_emotion', True) else '無効'}
・構造語除外: {'有効' if self.config.get('exclude_structural_words', True) else '無効'}
・厳密な品詞フィルタリング: {'有効' if self.config.get('strict_pos_filtering', True) else '無効'}

■ 除外語統計
・推論・感情語: {len(self.inference_emotion_words)}語
  例: {', '.join(list(self.inference_emotion_words)[:10])}...

・構造語: {len(self.structural_words)}語
  例: {', '.join(list(self.structural_words)[:10])}...

・機能語: {len(self.functional_words)}語
  例: {', '.join(list(self.functional_words)[:10])}...

・敬称・敬語: {len(self.honorific_words)}語
  例: {', '.join(list(self.honorific_words)[:10])}...

・総除外語数: {len(self.all_excluded_words)}語

■ 特別強化されたフィルタリング
・敬称の完全除去: 「さん」「ちゃん」「くん」「様」等の確実な除外
・様態表現の除去: 「よう」「ような」「みたい」「らしい」等の包括的除外
・語尾パターンマッチング: より確実な除外メカニズム

■ フィルタリング効果
この設定により、分析の焦点は以下の語彙タイプに絞られます：
・実質的な意味を持つ名詞（一般名詞、専門用語）
・重要な動作を表す動詞
・特徴的な形容詞
・固有名詞（人名・組織名・敬称は除外）

これにより、テキストの核心的な内容がより明確に浮かび上がります。
        """
        return report.strip()
    
    def create_interactive_network(self, pair_counter, output_path):
        """インタラクティブなネットワーク図の作成"""
        # NetworkXグラフの構築
        G = nx.Graph()
        top_pairs = pair_counter.most_common(self.config['network_top_n'])
        
        for (w1, w2), weight in top_pairs:
            G.add_edge(w1, w2, weight=weight)
        
        # レイアウトの計算
        layouts = {
            'spring': nx.spring_layout(G, k=3, iterations=50),
            'kamada_kawai': nx.kamada_kawai_layout(G),
            'circular': nx.circular_layout(G)
        }
        
        # 最も分散の大きいレイアウトを選択
        best_layout = max(layouts.items(), 
                         key=lambda x: np.var([pos[0] for pos in x[1].values()]))
        pos = best_layout[1]
        
        # ノードの重要度計算
        centrality = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        pagerank = nx.pagerank(G)
        
        # Plotlyでのインタラクティブ可視化
        edge_x, edge_y = [], []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            weight = G[edge[0]][edge[1]]['weight']
            edge_info.append(f"{edge[0]} - {edge[1]}: {weight:.3f}")
        
        # エッジの描画
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # ノードの準備
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[node for node in G.nodes()],
            textposition="middle center",
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                reversescale=True,
                color=[],
                size=[],
                colorbar=dict(
                    thickness=15,
                    len=0.5,
                    x=1.02,
                    title="重要度"
                ),
                line=dict(width=2)
            )
        )
        
        # ノードの色とサイズを重要度に基づいて設定
        node_adjacencies = []
        node_sizes = []
        node_text = []
        
        for node in G.nodes():
            adjacencies = list(G.neighbors(node))
            node_adjacencies.append(len(adjacencies))
            
            # サイズは次数とPageRankの組み合わせ
            size = 20 + (centrality[node] * 40) + (pagerank[node] * 100)
            node_sizes.append(size)
            
            # ホバー情報
            info = (f"語: {node}<br>"
                   f"接続数: {len(adjacencies)}<br>"
                   f"次数中心性: {centrality[node]:.3f}<br>"
                   f"媒介中心性: {betweenness[node]:.3f}<br>"
                   f"PageRank: {pagerank[node]:.3f}")
            node_text.append(info)
        
        node_trace.marker.color = node_adjacencies
        node_trace.marker.size = node_sizes
        node_trace.hovertext = node_text
        
        # グラフの作成
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title={
                               'text': '語彙共起ネットワーク：高精度フィルタリング版',
                               'x': 0.5,
                               'font': {'size': 20}
                           },
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="推論・感情語・構造語を除外した、核心的な語彙関係を表示",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white'
                       ))
        
        # HTMLファイルとして保存
        html_path = output_path.replace('.png', '_interactive.html')
        fig.write_html(html_path)
        
        # 静的な画像も生成
        self._create_static_network(pair_counter, output_path)
        
        return html_path
    
    def _create_static_network(self, pair_counter, output_path):
        """静的ネットワーク図の生成"""
        G = nx.Graph()
        top_pairs = pair_counter.most_common(self.config['network_top_n'])
        
        for (w1, w2), weight in top_pairs:
            G.add_edge(w1, w2, weight=weight)
        
        if not G.edges():
            print("ネットワーク図を作成するのに十分な共起関係が見つかりませんでした。")
            return
        
        # 現代のmatplotlib対応
        fig, ax = plt.subplots(figsize=(16, 12))
        pos = nx.kamada_kawai_layout(G)
        
        # ノードの重要度計算
        centrality = nx.degree_centrality(G)
        
        # エッジの重み正規化
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        norm = Normalize(vmin=min(weights), vmax=max(weights))
        
        cmap = matplotlib.colormaps['Spectral']
        
        # ノードサイズ計算
        node_sizes = [500 + 1000 * centrality[node] for node in G.nodes()]
        
        # クリーンで読みやすい描画設定
        nx.draw_networkx_nodes(
            G, pos, 
            node_color='lightblue',
            node_size=node_sizes,
            alpha=0.8,
            edgecolors='lightgray',
            linewidths=0.5,
            ax=ax
        )
        
        nx.draw_networkx_edges(
            G, pos,
            width=[1 + 3 * norm(w) for w in weights],
            edge_color=[cmap(norm(w)) for w in weights],
            alpha=0.7,
            ax=ax
        )
        
        # フォントサイズを大きくして読みやすさを向上
        nx.draw_networkx_labels(
            G, pos,
            font_size=12,
            font_family='IPAGothic',
            font_weight='bold',
            ax=ax
        )
        
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_wordcloud(self, word_freq, output_path):
        """日本語対応のワードクラウド生成"""
        # 頻度辞書の準備
        filtered_freq = {word: freq for word, freq in word_freq.items() 
                        if freq >= self.config['min_frequency']}
        
        if not filtered_freq:
            return None
            
        # ワードクラウドの生成
        wordcloud = WordCloud(
            font_path='/usr/share/fonts/truetype/fonts-japanese-gothic.ttf',
            width=1200, height=800,
            background_color='white',
            max_words=100,
            colormap='viridis',
            relative_scaling=0.5,
            min_font_size=10
        ).generate_from_frequencies(filtered_freq)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def advanced_topic_modeling(self, texts):
        """改良されたトピックモデリング"""
        # テキストの前処理
        processed_docs = []
        for text in texts:
            words = self.enhanced_tokenize(text)
            processed_docs.append(' '.join(words))
        
        if not any(processed_docs):
            return None, None
            
        # TF-IDFベクトル化
        vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(processed_docs)
            feature_names = vectorizer.get_feature_names_out()
            
            # 最適なトピック数の決定
            best_topics = self._find_optimal_topics(processed_docs)
            
            # LDAモデルの訓練
            lda = LatentDirichletAllocation(
                n_components=best_topics,
                random_state=42,
                max_iter=100
            )
            
            lda.fit(tfidf_matrix)
            
            # トピックの抽出
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topic_weight = topic[top_words_idx]
                
                topic_info = {
                    'id': topic_idx,
                    'words': list(zip(top_words, topic_weight)),
                    'description': ' + '.join([f"{word}({weight:.3f})" 
                                             for word, weight in zip(top_words[:5], topic_weight[:5])])
                }
                topics.append(topic_info)
            
            return topics, lda
            
        except Exception as e:
            print(f"トピックモデリングエラー: {e}")
            return None, None
    
    def _find_optimal_topics(self, docs, max_topics=10):
        """最適なトピック数を見つける"""
        try:
            from gensim.corpora import Dictionary
            from gensim.models import LdaModel
            from gensim.models.coherencemodel import CoherenceModel
            
            # Gensimでのコヒーレンス計算
            texts = [doc.split() for doc in docs]
            dictionary = Dictionary(texts)
            corpus = [dictionary.doc2bow(text) for text in texts]
            
            coherence_scores = []
            for num_topics in range(2, min(max_topics + 1, len(docs))):
                lda_model = LdaModel(
                    corpus=corpus,
                    id2word=dictionary,
                    num_topics=num_topics,
                    random_state=42,
                    passes=10
                )
                
                coherence_model = CoherenceModel(
                    model=lda_model,
                    texts=texts,
                    dictionary=dictionary,
                    coherence='c_v'
                )
                
                coherence_scores.append(coherence_model.get_coherence())
            
            # 最高スコアのトピック数を返す
            best_num = range(2, min(max_topics + 1, len(docs)))[np.argmax(coherence_scores)]
            return best_num
            
        except:
            # エラーの場合はデフォルト値
            return min(self.config['topic_num'], len(docs) // 2)
    
    def create_analysis_dashboard(self, all_features, topics, output_dir):
        """分析結果のダッシュボード作成"""
        # 複数のサブプロットを含む総合ダッシュボード
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('語彙頻度分布', 'トピック分布', '文書統計', 'フィルタリング効果'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. 語彙頻度分布
        word_freq = Counter()
        for features in all_features:
            word_freq.update(features['word_frequency'])
        
        top_words = word_freq.most_common(20)
        if top_words:
            fig.add_trace(
                go.Bar(x=[w[0] for w in top_words], y=[w[1] for w in top_words],
                       name="語彙頻度", marker_color='skyblue'),
                row=1, col=1
            )
        
        # 2. トピック分布（円グラフ）
        if topics:
            topic_labels = [f"トピック{t['id']+1}" for t in topics]
            topic_values = [len(t['words']) for t in topics]
            
            fig.add_trace(
                go.Pie(labels=topic_labels, values=topic_values, name="トピック"),
                row=1, col=2
            )
        
        # 3. 文書統計（散布図）
        doc_stats = pd.DataFrame([
            {
                'TTR': f['ttr'],
                '語数': f['word_count'],
                '文字数': f['char_count'],
                '平均語長': f['avg_word_length']
            }
            for f in all_features
        ])
        
        if not doc_stats.empty:
            fig.add_trace(
                go.Scatter(
                    x=doc_stats['語数'], 
                    y=doc_stats['TTR'],
                    mode='markers',
                    marker=dict(
                        size=doc_stats['文字数']/100,
                        color=doc_stats['平均語長'],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    name="文書特性"
                ),
                row=2, col=1
            )
        
        # 4. フィルタリング効果
        filter_categories = ['推論・感情語', '構造語', '機能語', '敬称・敬語']
        filter_counts = [
            len(self.inference_emotion_words),
            len(self.structural_words), 
            len(self.functional_words),
            len(self.honorific_words)  # 新しいカテゴリ
        ]
        
        fig.add_trace(
            go.Bar(x=filter_categories, y=filter_counts,
                   name="除外語数", marker_color='coral'),
            row=2, col=2
        )
        
        # レイアウト調整
        fig.update_layout(
            height=800,
            title_text="テキストマイニング総合ダッシュボード（改良版フィルタリング）",
            title_x=0.5,
            showlegend=False
        )
        
        # HTMLとして保存
        dashboard_path = os.path.join(output_dir, 'analysis_dashboard.html')
        fig.write_html(dashboard_path)
        
        return dashboard_path
    
    def generate_comprehensive_report(self, all_features, topics, pair_counter):
        """包括的な分析レポートの生成（改良版）"""
        # 全体統計の計算
        total_words = sum(f['word_count'] for f in all_features)
        total_chars = sum(f['char_count'] for f in all_features)
        unique_words = len(set().union(*[f['words'] for f in all_features]))
        
        avg_ttr = np.mean([f['ttr'] for f in all_features])
        avg_word_length = np.mean([f['avg_word_length'] for f in all_features])
        
        # 最頻出語の分析
        all_word_freq = Counter()
        for features in all_features:
            all_word_freq.update(features['word_frequency'])
        
        top_words = all_word_freq.most_common(15)
        
        # 共起関係の分析
        top_cooccurrences = pair_counter.most_common(15)
        
        # トピック要約
        topic_summary = ""
        if topics:
            for i, topic in enumerate(topics):
                top_5_words = [word for word, _ in topic['words'][:5]]
                topic_summary += f"・トピック{i+1}: {' / '.join(top_5_words)}\n"
        
        # フィルタリング効果の統計
        filtering_stats = self.create_filtering_report()
        
        # レポート本文の生成
        report = f"""
【高度テキストマイニング包括分析レポート】
生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}

{filtering_stats}

■ 基本統計情報
・分析対象文書数: {len(all_features)}件
・総語数: {total_words:,}語（フィルタリング後）
・総文字数: {total_chars:,}文字
・ユニーク語数: {unique_words:,}語
・平均語彙豊富度(TTR): {avg_ttr:.3f}
・平均語長: {avg_word_length:.2f}文字

■ 重要語トップ15（フィルタリング適用後）
{chr(10).join([f'・{word}: {freq}回' for word, freq in top_words])}

■ 注目される共起関係トップ15
{chr(10).join([f'・「{w1}」と「{w2}」: 関連度{freq:.3f}' for (w1, w2), freq in top_cooccurrences])}

■ 発見されたトピック
{topic_summary}

■ 分析の洞察（改良版フィルタリング適用）
この高度フィルタリング分析により、以下のような特徴が明確になりました：

1. 語彙の質的分析: 推論・感情語と構造語を除外することで、テキストの核心的な内容語に焦点が当たりました。

2. 意味的中心性: 最頻出語「{top_words[0][0] if top_words else "N/A"}」は、テキストの実質的なテーマを表しています。

3. 概念的関連性: 共起分析により、意味的に重要な語彙間の関係が浮き彫りになりました。

4. テーマ構造: {len(topics) if topics else 0}個の明確なトピックが特定され、内容の構造化された理解が可能になりました。

■ フィルタリングの効果
・除去された語彙カテゴリにより、分析の精度が大幅に向上
・推論表現や構造語の除去により、実質的な内容に集中
・感情表現の除去により、客観的な内容分析が実現

■ 技術的詳細
・改良版形態素解析: Janome/MeCab + 言語学的フィルタリング
・除外語辞書: 推論・感情語、構造語、機能語の体系的分類
・可視化: インタラクティブネットワーク、改良版ワードクラウド
・分析手法: 共起ネットワーク、LDAトピックモデリング、TF-IDF

※ この分析は高度な自然言語処理技術と言語学的知見により生成されました。
※ フィルタリング設定により、テキストの本質的な内容構造が明確化されています。
        """
        
        return report.strip()
    
    def process_files(self):
        """メインの処理実行（改良版）"""
        print("=== 高度テキストマイニング分析開始 ===")
        print(f"フィルタリング設定: 推論・感情語除外={self.config.get('exclude_inference_emotion', True)}")
        print(f"                    構造語除外={self.config.get('exclude_structural_words', True)}")
        
        # ディレクトリの準備
        os.makedirs(self.config['archive_dir'], exist_ok=True)
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # ファイル処理
        all_features = []
        all_pair_counter = Counter()
        all_texts = []
        
        source_files = [f for f in os.listdir(self.config['source_dir']) 
                       if f.endswith('.txt')]
        
        if not source_files:
            print("処理対象のテキストファイルが見つかりません。")
            return
        
        print(f"{len(source_files)}個のファイルを処理中...")
        
        for file in source_files:
            file_path = os.path.join(self.config['source_dir'], file)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                if text.strip():
                    features = self.extract_enhanced_features(text)
                    all_features.append(features)
                    all_pair_counter.update(features['pairs'])
                    all_texts.append(text)
                
                # ファイルをアーカイブに移動
                shutil.move(file_path, os.path.join(self.config['archive_dir'], file))
                
            except Exception as e:
                print(f"ファイル{file}の処理中にエラー: {e}")
                continue
        
        if not all_features:
            print("処理可能なテキストデータがありませんでした。")
            return
        
        print("高度分析を実行中...")
        
        # 各種分析の実行
        try:
            # 1. ネットワーク分析とワードクラウド
            network_path = os.path.join(self.config['output_dir'], 'network_filtered.png')
            interactive_network = self.create_interactive_network(all_pair_counter, network_path)
            
            # 2. ワードクラウド
            all_word_freq = Counter()
            for features in all_features:
                all_word_freq.update(features['word_frequency'])
            
            wordcloud_path = os.path.join(self.config['output_dir'], 'wordcloud_filtered.png')
            self.create_wordcloud(all_word_freq, wordcloud_path)
            
            # 3. トピックモデリング
            topics, lda_model = self.advanced_topic_modeling(all_texts)
            
            # 4. ダッシュボード作成
            dashboard_path = self.create_analysis_dashboard(all_features, topics, self.config['output_dir'])
            
            # 5. 包括レポート生成
            report = self.generate_comprehensive_report(all_features, topics, all_pair_counter)
            
            # 結果の保存
            self.results = {
                'report': report,
                'network_path': network_path,
                'interactive_network': interactive_network,
                'wordcloud_path': wordcloud_path,
                'dashboard_path': dashboard_path,
                'topics': topics
            }
            
            # レポートファイルの保存
            report_path = os.path.join(self.config['output_dir'], 'analysis_report_filtered.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print("=== 分析完了！ ===")
            print(f"・ネットワーク図: {network_path}")
            print(f"・インタラクティブ版: {interactive_network}")
            print(f"・ワードクラウド: {wordcloud_path}")
            print(f"・ダッシュボード: {dashboard_path}")
            print(f"・レポート: {report_path}")
            
            # フィルタリング効果の表示
            total_excluded = len(self.all_excluded_words)
            print(f"\n📊 フィルタリング統計:")
            print(f"・総除外語数: {total_excluded}語")
            print(f"・推論・感情語: {len(self.inference_emotion_words)}語")
            print(f"・構造語: {len(self.structural_words)}語")
            print(f"・機能語: {len(self.functional_words)}語")
            print(f"・敬称・敬語: {len(self.honorific_words)}語")
            print(f"\n✅ 特別強化: 「さん」「よう」等の確実な除外を実装")
            
            # メール送信
            self.send_enhanced_email()
            
        except Exception as e:
            print(f"分析中にエラーが発生: {e}")
            raise
    
    def send_enhanced_email(self):
        """改良されたメール送信機能"""
        if not self.results:
            print("送信するデータがありません。")
            return
        
        try:
            msg = MIMEMultipart()
            msg['Subject'] = "高度テキストマイニング分析レポート（改良版フィルタリング）"
            msg['From'] = self.config['from_email']
            msg['To'] = self.config['to_email']
            
            # レポート本文
            msg.attach(MIMEText(self.results['report'], 'plain', 'utf-8'))
            
            # 添付ファイル
            attachments = [
                ('network_filtered.png', self.results['network_path']),
                ('wordcloud_filtered.png', self.results['wordcloud_path'])
            ]
            
            for filename, filepath in attachments:
                if os.path.exists(filepath):
                    with open(filepath, 'rb') as f:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(f.read())
                        encoders.encode_base64(part)
                        part.add_header('Content-Disposition', f'attachment; filename="{filename}"')
                        msg.attach(part)
            
            # 送信
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(self.config['from_email'], self.config['app_password'])
                smtp.send_message(msg)
            
            print("メール送信完了！")
            
        except Exception as e:
            print(f"メール送信エラー: {e}")

# 実行部分
if __name__ == "__main__":
    miner = AdvancedTextMiner()
    miner.process_files()
