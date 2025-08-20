import tavr
from token_replacement_options import get_replacement_options

import os
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# TODO: ВЫНЕСТИ В ОТДЕЛЬНЫЙ КОНФИГ ИЛИ В ПЕРЕМЕННУЮ ОКРУЖЕНИЯ ИЛИ СДЕЛАТЬ ЧЕРЕЗ ДОКЕР ПУТИ
current_directory = os.getcwd()

app = FastAPI()

origins = ["http://127.0.0.1:8000", "http://localhost:8000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Essay(BaseModel):
    text: str
    
class TokenInfo(BaseModel):
    text: str
    level: Optional[str]
    pos: str
    
class Analytics(BaseModel):
    marked_up_tokens: Dict[int, TokenInfo]
    level: str
    
    stats_table: str
    trigrams_table: str
    academic_formulas_table: str
    academic_collocations_table: str
    academic_words_table: str
    recurring_lemmas_table: str
    
    len_academic_formulas_table: int
    len_academic_collocations_table: int
    len_academic_words_table: int
    recurring_lemmas: List[str]
    
class Replacements(BaseModel):
    lemmas: List[str]
    levels: List[str]
    

@app.post("/essay-analytics")
def analyze_essay(essay: Essay):
    text_analysis = tavr.TextAnalysis(essay.text)
    
    text_analysis.create_vocabulary_chart()
    text_analysis.create_vocabulary_chart(small=True)
    
    marked_up_tokens_raw = text_analysis.marked_up_tokens
    marked_up_tokens = dict()
    for i, (token, token_dict) in enumerate(marked_up_tokens_raw.items()):
        if token_dict["functional_word"] or token_dict["punct"]:
            token_info = TokenInfo(
                text=token.text,
                level=None,
                pos=token.pos_
            )
        else:
            token_info = TokenInfo(
                text=token.text,
                level=token_dict["level"],
                pos=token.pos_
            )
        marked_up_tokens[i] = token_info
    
    tables = {
        "trigrams_table": \
            text_analysis.get_trigrams_dataframe(entries_limit=10), 
        "academic_formulas_table": \
            text_analysis.get_academic_formulas_dataframe(entries_limit=5), 
        "academic_collocations_table": \
            text_analysis.get_academic_collocations_dataframe(entries_limit=5), 
        "academic_words_table": \
            text_analysis.get_academic_words_dataframe(entries_limit=6), 
        "recurring_lemmas_table": \
            text_analysis.get_recurring_lemmas_dataframe(entries_limit=10)
    }
    
    analytics = Analytics(
            marked_up_tokens=marked_up_tokens,
            level=text_analysis.get_level(),
            stats_table=text_analysis.get_stats_dataframe().to_html(index=False, header=False),
            trigrams_table=tables["trigrams_table"].to_html(index=False),
            academic_formulas_table=tables["academic_formulas_table"].to_html(index=False),
            academic_collocations_table=tables["academic_collocations_table"].to_html(index=False),
            academic_words_table=tables["academic_words_table"].to_html(index=False),
            recurring_lemmas_table=tables["recurring_lemmas_table"].to_html(index=False),
            len_academic_formulas_table=len(tables["academic_formulas_table"]),
            len_academic_collocations_table=len(tables["academic_collocations_table"]),
            len_academic_words_table=len(tables["academic_words_table"]),
            recurring_lemmas=tables["recurring_lemmas_table"]["Lemma"].tolist()
    )
       
    return analytics

@app.post("/replacements")
def get_replacements(token_info: TokenInfo):
    lemmas, levels = get_replacement_options(token_info.text, 
                                             token_info.pos)
    
    assert isinstance(lemmas, list)
    assert isinstance(levels, list)
    assert all(isinstance(lemma, str) for lemma in lemmas)
    assert all(isinstance(level, str) for level in lemmas)
    
    return Replacements(
        lemmas=lemmas,
        levels=levels
    )
    

app.mount("/tmp", 
          StaticFiles(directory=f"{current_directory}/tmp"), 
          name="tmp")
app.mount("/", 
          StaticFiles(directory=f"{current_directory}/static", html=True), 
          name="static")
    
if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)