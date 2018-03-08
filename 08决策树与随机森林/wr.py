def wr(R,v,C,m):
    """
        parameter:
            R:this movie average grade
            v:evaluated people
            C:all movie average grade
            m:the first 250 lowest scores
    """
    return (v/(v+m))*R +(m/(v+m))*C