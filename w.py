
if __name__ == '__main__':
    with open('t.txt', 'w') as f:
        for i in range(1, 551):
            f.writelines('%03d.png\n' % i)

