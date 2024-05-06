def get_point_distance(gs,start,line,distance,is_left):
            curr_distnace = 0
            point_before=None
            branched = False
            last_point=self.points[start]
            point=self.points[list(line[start])[0]]
            while curr_distnace<distance:
                new_curr_distnace=curr_distnace+np.linalg.norm(np.array(point)-np.array(last_point))
                if new_curr_distnace>distance:
                    t = (distance-curr_distnace)/np.linalg.norm(np.array(point)-np.array(last_point))
                    result = tuple(np.array(last_point)*(1-t)+t*np.array(point))
                    return result,last_point,branched
                else:
                    curr_distnace=new_curr_distnace


                point_before=last_point
                last_point=point
                points=[self.points[p] for p in line[self.points.index(last_point)] if self.points[p]!=point_before]
                if (len(points)==1):
                    point=points[0]
                elif (len(points)>1):
                    print(points,last_point)
                    s2=lambda x: s(x,last_point)
                    s_points = sorted(points,key=s2)
                    branched=True
                    if is_left:
                        point=s_points[-1]
                    else:
                        point=s_points[0]
                elif (len(points)==0):
                    return last_point,point_before,branched


            return last_point,point_before,branched

def s(a,b):
            v1 = np.array(a) - np.array(b)
            v1=v1/np.linalg.norm(v1)
            v1=v1+np.array(b)
            v2 = np.array(b) - self.center
            v1=v1/np.linalg.norm(v1)
            v2=v2/np.linalg.norm(v2)
            return np.cross(v1,v2)[1]



path=[822, 821, 820, 819, 818, 817, 816, 815, 814, 813, 812, 811, 810, 809, 808, 807, 806, 805, 804, 803, 802, 801, 800, 799, 798, 797, 796, 795, 794, 793, 792, 791, 790, 789, 788, 787, 786, 785, 784, 783, 782, 781, 780, 779, 778, 777, 776, 775, 774, 773]
star,end = (822, 773)
line_5=[(-392.3017272949219, -81.3869857788086, -19.975624084472656), (-389.3667907714844, -80.44689178466797, -20.268430709838867), (-386.4320068359375, -79.5067138671875, -20.53957748413086), (-383.4963073730469, -78.56597900390625, -20.777233123779297), (-380.5615539550781, -77.6256103515625, -21.010353088378906), (-377.6272277832031, -76.68504333496094, -21.20635986328125), (-374.6934509277344, -75.74459075927734, -21.388233184814453), (-371.7606201171875, -74.80402374267578, -21.534475326538086), (-368.8294677734375, -73.86150360107422, -21.47234344482422), (-365.9059143066406, -72.919677734375, -21.274620056152344), (-362.9896545410156, -71.97901153564453, -20.980775833129883), (-360.0786437988281, -71.03958892822266, -20.642433166503906), (-357.1708984375, -70.10089111328125, -20.274621963500977), (-354.2653503417969, -69.16292572021484, -19.905006408691406), (-351.360595703125, -68.22531127929688, -19.535783767700195), (-348.45611572265625, -67.28793334960938, -19.175771713256836), (-345.5529479980469, -66.35114288330078, -18.82306480407715), (-342.64892578125, -65.41432189941406, -18.48641586303711), (-339.7460021972656, -64.47795104980469, -18.161598205566406), (-336.8433532714844, -63.541908264160156, -17.84564781188965), (-333.94140625, -62.606117248535156, -17.537033081054688), (-331.03045654296875, -61.669342041015625, -17.406391143798828), (-328.1173400878906, -60.73286819458008, -17.38176727294922), (-325.2058410644531, -59.797245025634766, -17.39495277404785), (-322.2958068847656, -58.862152099609375, -17.40873146057129), (-319.38714599609375, -57.92717742919922, -17.401714324951172), (-316.4805908203125, -56.99274444580078, -17.362268447875977), (-313.5763854980469, -56.058650970458984, -17.292713165283203), (-310.675048828125, -55.1251335144043, -17.18702507019043), (-307.7760925292969, -54.191993713378906, -17.042531967163086), (-304.8800048828125, -53.25956344604492, -16.868803024291992), (-301.9867858886719, -52.32776641845703, -16.683025360107422), (-299.0944519042969, -51.3963737487793, -16.499752044677734), (-296.20361328125, -50.46562576293945, -16.32334327697754), (-293.3138122558594, -49.53519821166992, -16.158248901367188), (-290.42510986328125, -48.60529327392578, -16.00498390197754), (-287.5372009277344, -47.67581558227539, -15.865565299987793), (-284.65087890625, -46.74684143066406, -15.732921600341797), (-281.7655029296875, -45.81831741333008, -15.611183166503906), (-278.8816833496094, -44.89038848876953, -15.504253387451172), (-275.9986267089844, -43.963008880615234, -15.414502143859863), (-273.1174011230469, -43.036109924316406, -15.325437545776367), (-270.2379455566406, -42.109798431396484, -15.240067481994629), (-267.360107421875, -41.18406295776367, -15.159944534301758), (-264.4837341308594, -40.25887680053711, -15.086181640625), (-261.60906982421875, -39.33431625366211, -15.01974868774414), (-258.73583984375, -38.410335540771484, -14.962925910949707), (-255.86447143554688, -37.48703384399414, -14.916769981384277), (-252.99440002441406, -36.564273834228516, -14.884980201721191), (-250.1262664794922, -35.642311096191406, -14.868719100952148), (-247.2594757080078, -34.72101593017578, -14.875179290771484), (-244.3946990966797, -33.80073928833008, -14.913866996765137), (-241.53163146972656, -32.8812370300293, -14.981366157531738), (-238.6708221435547, -31.962879180908203, -15.076955795288086), (-235.81280517578125, -31.045696258544922, -15.201559066772461), (-232.9574737548828, -30.13008689880371, -15.375199317932129), (-230.10699462890625, -29.216886520385742, -15.617944717407227), (-227.2671356201172, -28.30906105041504, -16.000986099243164), (-224.43478393554688, -27.40489959716797, -16.448196411132812), (-221.60549926757812, -26.502443313598633, -16.912195205688477), (-218.77713012695312, -25.600656509399414, -17.3636474609375), (-215.94898986816406, -24.69923973083496, -17.799840927124023), (-213.122314453125, -23.798809051513672, -18.232627868652344), (-210.29527282714844, -22.89832878112793, -18.63875389099121), (-207.46852111816406, -21.99806785583496, -19.023630142211914), (-204.64173889160156, -21.09781837463379, -19.38298797607422), (-201.8150177001953, -20.197433471679688, -19.710710525512695), (-198.98851013183594, -19.29690933227539, -20.008647918701172), (-196.1641387939453, -18.397525787353516, -20.307300567626953), (-193.341796875, -17.499183654785156, -20.603132247924805), (-190.5222930908203, -16.602554321289062, -20.912899017333984), (-187.70582580566406, -15.70773696899414, -21.23538589477539), (-184.8930206298828, -14.81539535522461, -21.585311889648438), (-182.08404541015625, -13.92566204071045, -21.961673736572266), (-179.2808380126953, -13.039889335632324, -22.384458541870117), (-176.48367309570312, -12.158421516418457, -22.854202270507812), (-173.69175720214844, -11.280838966369629, -23.358436584472656), (-170.90382385253906, -10.406401634216309, -23.88018035888672), (-168.1181182861328, -9.534253120422363, -24.40662384033203), (-165.33340454101562, -8.663576126098633, -24.924335479736328), (-162.5489501953125, -7.793887615203857, -25.425148010253906), (-159.76368713378906, -6.924283504486084, -25.896188735961914), (-156.97752380371094, -6.054423809051514, -26.331666946411133), (-154.1912078857422, -5.184622764587402, -26.73627281188965), (-151.40481567382812, -4.314601898193359, -27.10487937927246), (-148.61888122558594, -3.4446661472320557, -27.441999435424805), (-145.8340301513672, -2.5749831199645996, -27.75018882751465), (-143.0504150390625, -1.7056363821029663, -28.02964973449707), (-140.2691650390625, -0.8376744985580444, -28.296621322631836), (-137.48971557617188, 0.029769837856292725, -28.536823272705078), (-134.7122802734375, 0.897530198097229, -28.736270904541016), (-131.9375457763672, 1.7662893533706665, -28.882213592529297), (-129.1667022705078, 2.636563539505005, -28.96339988708496), (-126.40158081054688, 3.5086324214935303, -28.970157623291016), (-123.64590454101562, 4.383413791656494, -28.874494552612305), (-120.90492248535156, 5.260860919952393, -28.657365798950195), (-118.1855697631836, 6.1402459144592285, -28.301734924316406), (-115.47783660888672, 7.019129276275635, -27.896366119384766), (-112.78778839111328, 7.897517204284668, -27.413597106933594), (-110.11882781982422, 8.774903297424316, -26.845905303955078), (-107.47193145751953, 9.650575637817383, -26.199230194091797), (-104.84599304199219, 10.524016380310059, -25.48567008972168), (-102.23828125, 11.394969940185547, -24.720340728759766), (-99.64523315429688, 12.26339054107666, -23.919206619262695), (-97.06283569335938, 13.12944507598877, -23.097440719604492), (-94.4872817993164, 13.993393898010254, -22.26854133605957), (-91.91546630859375, 14.855461120605469, -21.44384002685547), (-89.34484100341797, 15.715897560119629, -20.63226318359375), (-86.77381134033203, 16.574832916259766, -19.840452194213867), (-84.2013931274414, 17.43236541748047, -19.072906494140625), (-81.62727355957031, 18.288503646850586, -18.332216262817383), (-79.05162811279297, 19.143198013305664, -17.619367599487305)]
line_8=[((-146.35, -2.85, 13.97), (-150.0, -3.98, 13.6)), ((-150.0, -3.98, 13.6), (-153.71, -5.12, 12.73)), ((-153.71, -5.12, 12.73), (-161.21, -7.43, 11.03)), ((-161.21, -7.43, 11.03), (-165.0, -8.61, 10.39)), ((-165.0, -8.61, 10.39), (-172.55, -11.03, 10.64)), ((-172.55, -11.03, 10.64), (-176.08, -12.23, 12.45)), ((-176.08, -12.23, 12.45), (-179.55, -13.42, 14.41)), ((-179.55, -13.42, 14.41), (-182.96, -14.6, 16.52)), ((-182.96, -14.6, 16.52), (-189.56, -16.93, 21.18)), ((-189.56, -16.93, 21.18), (-192.75, -18.09, 23.72)), ((-192.75, -18.09, 23.72), (-210.51, -24.85, 40.89)), ((-210.51, -24.85, 40.89), (-213.28, -25.97, 43.94)), ((-213.28, -25.97, 43.94), (-216.01, -27.07, 47.01)), ((-216.01, -27.07, 47.01), (-218.7, -28.18, 50.09)), ((-218.7, -28.18, 50.09), (-224.05, -30.39, 56.19)), ((-224.05, -30.39, 56.19), (-229.42, -32.62, 62.04)), ((-229.42, -32.62, 62.04), (-237.65, -35.98, 70.22)), ((-237.65, -35.98, 70.22), (-243.27, -38.24, 75.26)), ((-243.27, -38.24, 75.26), (-251.92, -41.63, 82.25)), ((-251.92, -41.63, 82.25), (-254.86, -42.77, 84.42)), ((-254.86, -42.77, 84.42), (-257.84, -43.91, 86.5)), ((-257.84, -43.91, 86.5), (-263.92, -46.18, 90.36)), ((-263.92, -46.18, 90.36), (-267.04, -47.32, 92.1)), ((-267.04, -47.32, 92.1), (-270.22, -48.46, 93.68)), ((-270.22, -48.46, 93.68), (-276.84, -50.74, 96.21)), ((-276.84, -50.74, 96.21), (-280.29, -51.87, 97.06)), ((-280.29, -51.87, 97.06), (-283.85, -53.0, 97.6)), ((-283.85, -53.0, 97.6), (-287.48, -54.11, 97.84)), ((-287.48, -54.11, 97.84), (-294.86, -56.31, 97.75)), ((-294.86, -56.31, 97.75), (-298.61, -57.4, 97.43)), ((-298.61, -57.4, 97.43), (-302.37, -58.48, 96.94)), ((-302.37, -58.48, 96.94), (-309.98, -60.58, 95.13)), ((-309.98, -60.58, 95.13), (-313.79, -61.6, 93.85)), ((-313.79, -61.6, 93.85), (-317.61, -62.6, 92.32)), ((-317.61, -62.6, 92.32), (-325.18, -64.52, 88.43)), ((-325.18, -64.52, 88.43), (-328.91, -65.44, 86.07)), ((-328.91, -65.44, 86.07), (-332.58, -66.33, 83.38)), ((-332.58, -66.33, 83.38), (-339.79, -68.07, 77.77)), ((-339.79, -68.07, 77.77), (-343.32, -68.93, 74.82)), ((-343.32, -68.93, 74.82), (-357.0, -72.3, 62.17)), ((-357.0, -72.3, 62.17), (-360.31, -73.12, 58.83)), ((-360.31, -73.12, 58.83), (-363.56, -73.95, 55.45)), ((-363.56, -73.95, 55.45), (-369.97, -75.6, 48.57)), ((-369.97, -75.6, 48.57), (-373.13, -76.44, 45.14)), ((-373.13, -76.44, 45.14), (-379.48, -78.17, 38.57)), ((-379.48, -78.17, 38.57), (-382.74, -79.09, 35.68)), ((-382.74, -79.09, 35.68), (-386.11, -80.08, 33.34)), ((-386.11, -80.08, 33.34), (-389.46, -81.07, 30.95)), ((-389.46, -81.07, 30.95), (-396.08, -83.03, 26.1))]