import re


class Splitter:
    """
    helper class to split text into chunks, offers different splitting strategies
    stragies:
        - split by chunksize, starting at index 0
        - split around delimiter, with a maximum of chunksize characters around it
    """

    # splits a string into chunks of a given size
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
    # specified by beforeLength and afterLength
    # text: text to split
    # beforeLength: number of characters before the delimiter token
    # afterLength: number of characters after the delimiter token
    # delimiter: delimiter token, defaults to <mask>
    @classmethod
    def split_around_mask(self, text, beforeLength, afterLength, delimiter=r'/<mask>/'):
        # find all masks using regex
        maskIndices = [m.start() for m in re.finditer(delimiter, text)]
        for maskIndex in maskIndices:
            # get the lengths of the text before and after the mask constrained by text boundaries
            beforeIndex = max(0, maskIndex-beforeLength)
            afterIndex = min(len(text) - 1, maskIndex + afterLength)
            yield text[beforeIndex:afterIndex]

    # TODO: implement
    # takes sentence endings into account, preventing sentences from being split, which might make
    # the prediction of some masks more difficult.
    @classmethod
    def split_by_sentences(self, text):
        pass
