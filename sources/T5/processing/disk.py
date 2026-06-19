from datetime import datetime
import pandas as pd
class InMemoryFile:
    def __init__(self, filename: str, text: str, txtclass: str):
        self.filename = filename
        self.text = text
        self.txtclass = txtclass

    @property
    def char_count(self) -> int:
        return len(self.text)

    def __len__(self) -> int:
        return len(self.text)

    def __str__(self, preview_len: int = 50) -> str:
        preview = self.text[:preview_len]
        return (
            f"{self.filename} - {self.txtclass} len: {self.char_count}| text: {preview}"
        )


def split_text_to_parts(source: InMemoryFile, part_len: int, parts_count: int) -> list[InMemoryFile]:
    result = []
    text = source.text
    for i in range(parts_count):
        start = i * part_len
        if start >= len(text):
            break

        chunk = text[start:start + part_len]
        result.append(
            InMemoryFile(
                filename=source.filename,
                text=chunk,
                txtclass=source.txtclass,
            )
        )

    return result

def load_inputs(path: str ='test_input'):
    # Чтение файлов для дообучения
    hdd_texts = []
    for i in range(6, 17):
        load_path = f'{path}/HDD/HDD0{i}.txt' if i < 10 else f'{path}/HDD/HDD{i}.txt'
        with open(load_path, 'r', encoding='utf-8') as f:
            text = f.read()
            print(f'{load_path} loaded;   symbols count: {len(text)}')
            hdd_texts.append(text)
    hdd_texts = [[f'тип диска: {val}', 'hdd'] for val in hdd_texts]


    ssd_texts = []
    for i in range(7, 16):
        load_path = f'{path}/SDD/SDD0{i}.txt' if i < 10 else f'{path}/SDD/SDD{i}.txt'
        with open(load_path, 'r', encoding='utf-8') as f:
            text = f.read()
            print(f'{load_path} loaded;   symbols count: {len(text)}')
            ssd_texts.append(text)
    ssd_texts = [[f'тип диска: {val}', 'ssd'] for val in ssd_texts]
    pairs = hdd_texts + ssd_texts
    #for pair in pairs:
    #    print(f'{pair[0][:50]} - {pair[1]}')


def load_semple():
    pairs = [
        ['тип диска: SSD (Solid State Drive) — твердотельный накопитель на флеш-памяти. Скорость чтения до 7000 МБ/с, без движущихся частей, бесшумный.',
         'ssd'],
        ['тип диска: SSD обеспечивает мгновенную загрузку ОС (10-15 сек) против 1+ минуты на HDD. Идеален для системы и программ.',
         'ssd'],
        ['тип диска: SSD работает при температурах до 70°C без перегрева. Потребляет 2-5 Вт против 6-10 Вт у HDD.',
         'ssd'],
        ['тип диска: SSD устойчив к ударам и вибрациям — нет механики. MTBF 1.5-2 млн часов.', 'ssd'],
        ['тип диска: NVMe SSD через PCIe 4.0 дает 7000/5000 МБ/с. SATA SSD — 550/500 МБ/с.', 'ssd'],
        ['тип диска: SSD использует TLC/QLC NAND. Ресурс записи 300-600 TBW на 1 ТБ.', 'ssd'],
        ['тип диска: SSD мгновенный Random 4K IOPS — до 1 млн. HDD — 100-200 IOPS.', 'ssd'],
        ['тип диска: SSD без seek time. Доступ к данным <0.1 мс против 8-12 мс у HDD.', 'ssd'],
        ['тип диска: SSD TRIM оптимизирует производительность. Удаляет неиспользуемые блоки.', 'ssd'],
        ['тип диска: SSD контроллеры Phison/Samsung обеспечивают кэш SLC и защиту от износа.', 'ssd'],
        ['тип диска: SSD PCIe 5.0 достигает 14000 МБ/с. Будущее для рабочих станций.', 'ssd'],
        ['тип диска: SSD энергосбережение 80% при простое. Идеален для ноутбуков.', 'ssd'],
        ['тип диска: SSD нечувствителен к ориентации. Работает вверх ногами/на боку.', 'ssd'],
        ['тип диска: SSD цена за ГБ падает. 1 ТБ SSD = 4000 руб против 5000 руб в 2023.', 'ssd'],
        ['тип диска: SSD для баз данных и виртуализации. Случайный доступ критичен.', 'ssd'],
        ['тип диска: SSD DRAM-кэш ускоряет mapping table. Без DRAM — медленнее.', 'ssd'],
        ['тип диска: SSD Wear Leveling равномерно распределяет износ ячеек NAND.', 'ssd'],
        ['тип диска: SSD Over-Provisioning 7-25% увеличивает долговечность.', 'ssd'],
        ['тип диска: SSD M.2 форм-фактор 2280/2230. Совместим с материнскими платами.', 'ssd'],
        ['тип диска: SSD для геймеров — текстуры/миры загружаются за 2-3 сек.', 'ssd'],
        ['тип диска: SSD с heatsink для PS5 и RTX 4090. Температура <60°C под нагрузкой.', 'ssd'],
        ['тип диска: SSD Power Loss Protection сохраняет данные при отключении питания.', 'ssd'],
        ['тип диска: SSD эмулирует NCQ до 32 команд одновременно.', 'ssd'],
        ['тип диска: SSD работает при -40..+85°C. Сертификация для промышленности.', 'ssd'],
        ['тип диска: SSD корпоративный класс. 5 лет гарантии, 1 DWPD.', 'ssd'],

        ['тип диска: HDD (Hard Disk Drive) — механический диск с магнитными пластинами. Объем до 24 ТБ по цене 15-20 руб/ГБ.',
         'hdd'],
        ['тип диска: HDD скорость чтения 100-200 МБ/с. Зависит от положения головки и 5400-7200 об/мин.', 'hdd'],
        ['тип диска: HDD для архивов и холодных данных. 10 ТБ HDD стоит как 1 ТБ SSD.', 'hdd'],
        ['тип диска: HDD шумит (30-40 дБ), греется (40-50°C). Подходит для NAS с RAID.', 'hdd'],
        ['тип диска: HDD интерфейсы SATA III (6 Гбит/с). Кэш 64-256 МБ ускоряет случайный доступ.', 'hdd'],
        ['тип диска: HDD CMR (Conventional Magnetic Recording) для серверов. SMR медленнее при записи.', 'hdd'],
        ['тип диска: HDD 7200 RPM enterprise класс. 550 ТБ/год нагрузка, 2.5 млн часов MTBF.', 'hdd'],
        ['тип диска: HDD гелиевые диски 20-24 ТБ. Герметичный корпус, меньше турбулентности.', 'hdd'],
        ['тип диска: HDD 512e/4Kn эмуляция секторов. Совместимость с legacy системами.', 'hdd'],
        ['тип диска: HDD для медиа-серверов Plex. Последовательная запись 200+ МБ/с.', 'hdd'],
        ['тип диска: HDD RAID5/6 — 20-30 дисков без потери производительности.', 'hdd'],
        ['тип диска: HDD TLER (Time Limited Error Recovery). 7 сек вместо 40 сек для RAID.', 'hdd'],
        ['тип диска: HDD 3.5" 26 ТБ Seagate Exos. HAMR технология на горизонте.', 'hdd'],
        ['тип диска: HDD NAS-диски WD Red/CMR. 180 ТБ/год для 24/7 работы.', 'hdd'],
        ['тип диска: HDD Vibration Protection — датчики корректируют положение головок.', 'hdd'],
        ['тип диска: HDD AR (All Frame) — многосекторная обработка для скорости.', 'hdd'],
        ['тип диска: HDD 5400 RPM для ноутбуков. Тишина и энергоэффективность.', 'hdd'],
        ['тип диска: HDD внешние USB 3.0/Thunderbolt. До 20 ТБ портативные боксы.', 'hdd'],
        ['тип диска: HDD для бэкапа. DuraWrite технология продлевает срок службы.', 'hdd'],
        ['тип диска: HDD 5 лет гарантии enterprise. Самозатвердевающий герметик.', 'hdd'],
        ['тип диска: HDD Multi-Rate — автоопределение SATA/SAS интерфейсов.', 'hdd'],
        ['тип диска: HDD PowerChoice — временное отключение головок для энергосбережения.', 'hdd'],
        ['тип диска: HDD Rotational Vibration Sensors для стоек серверов.', 'hdd'],
        ['тип диска: HDD Data Lifeguard — самодиагностика и пересчет секторов.', 'hdd'],
        ['тип диска: HDD 30 ТБ HAMR диски в 2026. Плотность 2.2 Тб/дюйм.', 'hdd']
    ]
    return pairs


from pathlib import Path

# читает все txt файлы в папке
def read_txt_files(folder_path):
    print(f'read: {folder_path}')
    folder = Path(folder_path)
    contents = []
    names = []
    txtClass = []
    predictedClass = []

    # обходит все .txt файлы
    for file_path in sorted(folder.rglob("*.txt")):
        if not file_path.is_file():
            continue

        text = file_path.read_text(encoding="utf-8")
        #print(text)
        name = file_path.name

        if name.startswith("hdd"):
            short_name = "hdd"
        elif name.startswith("ssd"):
            short_name = "ssd"
        else:
            short_name = ''.join(char for char in file_path.stem if not char.isdigit())   # если вдруг другое имя

        contents.append(text)
        names.append(name)
        txtClass.append(short_name.lower())
        predictedClass.append(None)

    result_df = pd.DataFrame({
    "filename": names,
    "text": contents,
    "label": txtClass,
    "predictedClass": predictedClass
    })
    return result_df



def load_input_txt(path: str ='train_input'):
    # Чтение файлов для дообучения
    hdd_texts = []
    for i in range(1, 6):
        with open(f'{path}/HDD/HDD0{i}.txt', 'r', encoding='utf-8') as f:
            text = f.read()
            print(f'{path}/HDD/HDD0{i}.txt loaded;   symbols count: {len(text)}')
            hdd_texts.append(text)
    hdd_texts = [[f'тип диска: {val}', 'hdd'] for val in hdd_texts]

    ssd_texts = []
    for i in range(1, 6):
        with open(f'{path}/SDD/SDD0{i}.txt', 'r', encoding='utf-8') as f:
            text = f.read()
            print(f'{path}/SDD/SDD0{i}.txt loaded;   symbols count: {len(text)}')
            ssd_texts.append(text)
    ssd_texts = [[f'тип диска: {val}', 'ssd'] for val in ssd_texts]
    pairs = hdd_texts + ssd_texts
    #for pair in pairs:
    #    print(f'{pair[0][:50]} - {pair[1]}')

    return pairs

def load_test_input_txt(path: str ='test_input'):
    # Чтение файлов для дообучения
    hdd_texts = []
    for i in range(6, 17):
        load_path = f'{path}/HDD/HDD0{i}.txt' if i < 10 else f'{path}/HDD/HDD{i}.txt'
        with open(load_path, 'r', encoding='utf-8') as f:
            text = f.read()
            print(f'{load_path} loaded;   symbols count: {len(text)}')
            hdd_texts.append(text)
    hdd_texts = [[f'тип диска: {val}', 'hdd'] for val in hdd_texts]


    ssd_texts = []
    for i in range(7, 16):
        load_path = f'{path}/SDD/SDD0{i}.txt' if i < 10 else f'{path}/SDD/SDD{i}.txt'
        with open(load_path, 'r', encoding='utf-8') as f:
            text = f.read()
            print(f'{load_path} loaded;   symbols count: {len(text)}')
            ssd_texts.append(text)
    ssd_texts = [[f'тип диска: {val}', 'ssd'] for val in ssd_texts]
    pairs = hdd_texts + ssd_texts
    #for pair in pairs:
    #    print(f'{pair[0][:50]} - {pair[1]}')

    return pairs

def chop_texts2(pairs: list, symbols_count: int, max_batches):
    answer = []
    for pair in pairs:
        batch_texts = []
        batch_labels = []
        for i in range(0, len(pair[0]), symbols_count):
            batch_texts.append([0][i:i + symbols_count])

        if len(batch_texts) < max_batches and max_batches > 0:
            batch_texts = batch_texts[:max_batches]

        answer.append([text for text in batch_texts])


def chop_texts(pairs: list, symbols_count: int, max_batches):
    answer = []
    for pair in pairs:
        batch_texts = []
        for i in range(0, len(pair[0]), symbols_count):
            batch_texts.append(pair[0][i:i + symbols_count])

        if len(batch_texts) > max_batches and max_batches > 0:
            batch_texts = batch_texts[0:max_batches]

        for text in batch_texts:
            newpair = [text, pair[1]]
            #print(f'label {pair[1]}')
            answer.append(newpair)
            #print(newpair)

    return answer


def write_txt_log(text: str, path: str = 'log.txt'):
    with open(path, 'a', encoding='utf-8') as file:
        file.write(f'[{datetime.now().strftime("%d.%m.%Y %H:%M:%S")}] \t{text}\n')