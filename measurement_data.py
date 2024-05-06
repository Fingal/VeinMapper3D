#from typing import Container

from io import TextIOWrapper


class ConcentrationData():
    def __init__(self,name_first,d_7_avg,d_15_avg,h_avg,d_7,d_15,h,radius) -> None:
        self.data = [name_first,d_7_avg,d_15_avg,h_avg,d_7,d_15,h,radius]

class ConcentrationExporter():
    def __init__(self) -> None:
        self.rows : list[ConcentrationData] = []

    def add(self,name_first,d_7_avg,d_15_avg,h_avg,d_7,d_15,h,radius) -> None:
        self.rows.append(ConcentrationData(name_first,d_7_avg,d_15_avg,h_avg,d_7,d_15,h,radius))

    def export(self,file: TextIOWrapper):
        if self.rows:
            file.write("label\td_7_avg\td_15_avg\th_avg\td_7\td_15\th\tradius\n")
            for row in self.rows:
                file.write('\t'.join(map(str,row.data))+'\n')
            
            self.rows = []
        


class AngleData():
    def __init__(self,name_first,name_second,angle) -> None:
        self.data = [name_first,name_second,angle]

class LineLineDistanceData():
    def __init__(self,name_first,name_second,distance) -> None:
        self.data = [name_first,name_second,distance]

    
class AngleExporter():
    def __init__(self) -> None:
        self.rows : list[AngleData] = []

    def add(self,name_first,name_second,angle) -> None:
        self.rows.append(AngleData(name_first,name_second,angle))

    def export(self,file: TextIOWrapper):
        if self.rows:
            file.write("first label\tsecond label\tangle\n")
            for row in self.rows:
                file.write('\t'.join(map(str,row.data))+'\n')

            self.rows = []

class LineLineDistanceExporter():
    def __init__(self) -> None:
        self.rows : list[LineLineDistanceData] = []

    def add(self,name_first,name_second,distance) -> None:
        self.rows.append(LineLineDistanceData(name_first,name_second,distance))

    def export(self,file: TextIOWrapper):
        if self.rows:
            file.write("first label\tsecond label\tdistance\n")
            for row in self.rows:
                file.write('\t'.join(map(str,row.data))+'\n')

            self.rows = []


