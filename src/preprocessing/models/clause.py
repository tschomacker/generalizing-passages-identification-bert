class Clause:

    def __init__(self):
        self.POSSIBLE_TAGS = ['gi', 'comment', 'nfr']
        self.POSSIBLE_TAGS_SUBTAGS = {  'gi'        : ['ALL','BARE','DIV','EXIST','MEIST','NEG'], 
                                        'comment'   : ['Einstellung','Interpretation','Meta'], 
                                        'nfr'       : ['Nichtfiktional','Nichtfiktional+mK'], 
                                        'nfr_ex_mk'       : ['Nichtfiktional']}
        self.tokens = []
        self.contextualized_text = None
        self.clause_text_embed_in_passage = None

    def __repr__(self):
        return self.text

    @property
    def text(self):
        """
        convert TokenList to human readable text
        """
        text = ''
        for i in range(0,len(self.tokens)):
            if 0 == i:
                text = self.tokens[i]['form']
            else:
                text += ' ' + self.tokens[i]['form']
            
        return self._clean_text(text)

    def _clean_text(self, text):
        text = text.replace('  ', '')
        #remove starting blank
        if ' ' == text[0]:
            text= text[1:]
        return text

    @property
    def subtags_raw(self):
        """
        Extracts the subtags of the passage from the features of the first token.
        There is no need to iterate over all tokens because each token in a partial sentence 
        should be labeled the same. For more info please the Annotationsrichtlinien or the 
        documentation of this project. This is in theory but there seems to be exception so
        i will iterate over all token

        return: list<str>
        """
        subtags = []
        for tag in self.POSSIBLE_TAGS:
            for token in self.tokens:
                if token.get(tag) is not None:
                    for subtag in token.get(tag):
                        # avoid duplicates
                        subtag = subtag.replace('[!]','')
                        if subtag not in subtags:
                            subtags.append(subtag)
        return subtags
    
    @property
    def subtags(self):
        """
        subtags without predecessing number
        """
        subtags = []
        #for i in range(0,len(self.subtags)):
        for subtag in self.subtags_raw:
            subtags.append(subtag.split(':')[1])
        return subtags

    @property
    def gi(self):
        """
        return one hot vector for gi subtags
        """
        return self._tag_vector('gi',False)

    @property
    def gi_none(self):
        """
        return one hot vector for gi subtags
        """
        return self._tag_vector('gi',True)

    @property
    def comment(self):
        """
        return one hot vector for comment subtags
        """
        return self._tag_vector('comment', False)

    @property
    def comment_none(self):
        """
        return one hot vector for comment subtags
        """
        return self._tag_vector('comment', True)

    @property
    def nfr(self):
        """
        return one hot vector for nfr subtags
        """
        return self._tag_vector('nfr', False)

    @property
    def nfr_none(self):
        """
        return one hot vector for nfr subtags
        """
        return self._tag_vector('nfr', True)

    @property
    def nfr_ex_mk(self):
        """
        Returns:
            str: one hot vector for nfr subtags
            but without the 'Nichtfiktional+mK' samples
        """
        return self._tag_vector('nfr_ex_mk', False)

    @property
    def generalization(self):
        """
        Returns:
            str: 0 when no gi-subtag is applied; 1 when at least one gi-subtag is applied
        """

        # invert the none-label -> 
        return str(1 if int(self._tag_vector('gi', True)[0]) == 0 else 0)

    @property
    def reflexive_ex_mk_binary(self):
        """
        Returns:
            str: 0 when no label is applied; 1 when at leat one label is applied
        """
        number_of_tags = (1 if int(self.comment_none[0]) == 0 else 0)+int(self.nfr_ex_mk)+int(self.generalization)
        return str(1 if number_of_tags > 0 else 0)
    
    

    @property
    def reflexive_ex_mk_multi(self):
        """
        Returns:
            str: one hot vector for 'gi'-'comment'-'nfr'
        """
        return self.generalization+str(1 if int(self.comment_none[0]) == 0 else 0)+self.nfr_ex_mk


    def _tag_vector(self, tag, add_none_label):
        """
        Args:
            tag (str) : tag-string
            add_none_label (boolean): Flag that indicates whether the vector should start with
            the none-label
        Returns:
            str: multi-hot vector for the subtags of the tag in string format
        """
        label = ''
        subtags = self.subtags
        #used for labeling entries without any label
        has_label = False
        
        for possible_subtag in self.POSSIBLE_TAGS_SUBTAGS[tag]:
            if possible_subtag in subtags:
                label += '1'
                has_label = True
            else:
                label += '0'
        if add_none_label:
            if has_label:
                label = '0'+label
            else:
                label = '1'+label
                if len(subtags) > 0:
                    self._tag_vector_sanity_check(subtags)
        return label
    
    def _tag_vector_sanity_check(self,subtags):
        """
        check for tags, that are unknown or not mapped properly
        """
        for assigned_subtag in subtags:
            found_it_in = ''
            for possible_tag in self.POSSIBLE_TAGS:
                for possible_subtag in self.POSSIBLE_TAGS_SUBTAGS[possible_tag]:
                    if assigned_subtag == possible_subtag:
                        found_it_in = possible_tag
            if found_it_in == '':
                print('unable to map:',assigned_subtag,'from\n', self.text)



    def append(self, token):
        self.tokens.append(token)
