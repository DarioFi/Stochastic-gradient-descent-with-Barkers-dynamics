class CircularList:
    # This class implements a rough Circular list tailored to do a Ruppert Polyak averaging
    def __init__(self, objs, size=None):
        if size is None and objs is None:
            raise Exception("Size and obj cannot be both None")
        if size is None and objs is not None:
            size = len(objs)
        if objs is not None:
            if len(objs) > size:
                raise Exception("Size too small")
            self._internal_list = list(objs)
        else:
            self._internal_list = []
            objs = []

        self._internal_list.extend([None] * (size - len(objs)))
        self.size = size
        self._pointer = 0

    def get_last(self):
        return self._internal_list[self._pointer]

    def rotate(self):
        self._pointer += 1
        self._pointer %= self.size

    def set_last(self, obj):
        self._internal_list[self._pointer] = obj

    @property
    def internal_list(self):
        return self._internal_list


if __name__ == '__main__':
    cq = CircularList([[1], [2], [3]], 3)
    print(cq.internal_list)

    print(cq.get_last())
    print(cq.get_last())
    cq.rotate()
    print(cq.get_last())
    cq.rotate()
    cq.rotate()
    print(cq.get_last())
    cq.rotate()
    print(cq.get_last())
    cq.rotate()
    x = cq.get_last()
    x.append(4)
    print(cq.get_last())