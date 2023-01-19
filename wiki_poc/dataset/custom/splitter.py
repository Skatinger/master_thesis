import re


class Splitter:
    """
    helper class to split text into chunks, offers different splitting strategies
    stragies:
        - split by chunksize, starting at index 0
        - split around delimiter, with a maximum of chunksize characters around it
    """

    # splits a string into chunks of a given size, e.g. with chunksize 1024 and batchSize 5
    # it returns an array of 5 strings, each with a length of 1024
    @classmethod
    def split_by_chunksize(self, text, chunksize, batchSize):
        fullLength = len(text)
        # iterate over starting indices in text for each batch
        for batchStartIndex in range(0, len(text), chunksize * batchSize):
            # iterate over starting indices for each chunk within the batch
            chunkBatch = []
            for chunkStartIndex in range(batchStartIndex, batchStartIndex + (chunksize * batchSize), chunksize):
                end = min(fullLength, chunkStartIndex + chunksize)
                chunkBatch.append(text[chunkStartIndex:end])
            yield chunkBatch

    # returns chunks of text evolving around a mask tokens, with text before and after the mask token
    # specified by beforeLength and afterLength. Any other mask tokens in the text are replaced with 'mask'
    # text: text to split
    # beforeLength: number of characters before the delimiter token
    # afterLength: number of characters after the delimiter token
    # delimiter: delimiter token, defaults to <mask>
    @classmethod
    def split_around_mask(self, text, beforeLength, afterLength, maskToken=r'<mask>'):
        # find all masks using regex
        maskIndices = [m.start() for m in re.finditer(maskToken, text)]
        maskLength = len(maskToken)
        for maskIndex in maskIndices:
            # get the lengths of the text before and after the mask constrained by text boundaries
            startIndex = max(0, maskIndex-beforeLength)
            endIndex = min(len(text) - 1, maskIndex + maskLength + afterLength)
            # move beforeIndex to start of word
            while (startIndex > 0 and text[startIndex] != ' '):
                startIndex -= 1
            # move afterIndex to end of word
            while (endIndex < len(text) - 1 and text[endIndex] != ' '):
                endIndex += 1
            # text before mask token with any other mask tokens removed
            before = re.sub(maskToken, 'mask', text[startIndex:maskIndex])
            # text after mask token with any other mask tokens removed
            after = re.sub(maskToken, 'mask', text[maskIndex + maskLength:endIndex])
            # yield the chunk of text
            yield before + maskToken + after

    # TODO: implement
    # takes sentence endings into account, preventing sentences from being split, which might make
    # the prediction of some masks more difficult.
    @classmethod
    def split_by_sentences(self, text):
        raise NotImplementedError
