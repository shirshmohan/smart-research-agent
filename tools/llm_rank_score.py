def llm_rank_score(title:str,snippet:str,url:str,llm):
    prompt=f"""Rate the credibility of the following source from 1(low) to 5(high):
Title:{title}
URL:{url}
Snippet:{snippet}
Return only a number from 1 to 5."""
    try:
        response = llm.invoke(prompt)
        return int(response.strip())
    except:
        return 2