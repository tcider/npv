# Библиотека для вычисляения Npv и др показателей
import numpy_financial as npf
# pip install numpy-financial

# Бибилотека оконных функций windows
import PySimpleGUI as sg
# pip install PySimpleGUI

# Библитеки вывода графиков
import matplotlib.pyplot as plt
# pip install matplotlib

# Библиотека для настройки оформления графиков
import seaborn as sns
# pip install seaborn


# Функция возвращающая пограничные значения NPV, IRR, PP, PI между классами "привлекателен" и "не привлекателен"
# В нашем случае это статические уровни, можно сказать выбранные экспертным методом.
# Их можно поменять в форме ввода инвестиционных данных
def get_class_levels():
    # Значения возваращются в виде словаря
    res = dict()
    res['NPV'] = 1500000
    res['IRR'] = 0.7
    res['PP'] = 1.33
    res['PI'] = 2.77
    return res


# По пограничным(пороговым) значениям NPV, IRR, PP, PI определяем коэффициенты (bi) канонической
# дискриминантной функции классификации D = b1*NPV + b2*IRR + b3*PP + b4*PI + b0,
# таким образом получаем реализацию дискриминантной функции
# Либо мы можем получить эти коэффциенты, построив машинную модель "обучения с учителем"
def get_koef(porog_value):
    # Берем обратное значение от порогового, тогда значение меньше порогового будут давать произведение b*X меньше 1
    return 1/porog_value


# Из полученных коэффициентов реализуем непосредственно саму дискриминантную функцию
def discriminant_function(class_levels, x_dict):

    # В качестве коэф-та b0 берем -2 тк у нас 3пороговых значения с обратными коэ-ми к-е дают произведение
    # при пороговом значение равным 1, а один коэф-т при значение PP имеет отрицательное значение,
    # поэтому для его "нейтрализации" нужно -1, итого 1+1+1-1 = 2.
    # Тогда дискриминантная функция будет возвращать положительно число(в тч 0)
    # если проект "подходит" и отрицательное если нет
    result = -2

    for key in x_dict:
        # В слачае с показателем срока окупаемости PP, он в отличии от других "чем меньше тем лучше"
        # поэтому его коэффициент отрицтельный
        if key == 'PP':
            koef = -get_koef(class_levels[key])
        else:
            koef = get_koef(class_levels[key])
        result += x_dict[key] * koef
    return result


# Вспомогательная функция проверки на пустоту введенных текстовых полей
def check_text_values(values):

    # Тк считываемые из формы значения помещаются в словарь то получаем доступ к ним через ключи
    for key in values:
        # Проверяем размер введенных данных
        if len(values[key]) == 0:
            # Если введено пустое поле тот возвращаем False
            #print(key, values[key]) #FIXME Дебаг
            return False
    # Если все ок то True
    return True


# Вспомогательная функция проверки на тип данных (дата и числовые) введенных в окне Инвест данные
# Кроме того функция сразу переводит значения в числовой тип(тк все вводимые данные изначально воспринимаются как тест)
def check_data_values(values):

    # В цикле проходим по всем зачениям (через ключи словаря) и конвертируем их в числовые если конвертация не возможна
    # Значит введенно значение не было чистым числом
    for key in values:
        # Значение словаря с ключом 'start' не конвертируем тк это дата
        if key == 'start':
            continue

        # Делаем конвертацию через механизм исключений в Питон, если будет ошибка знаяит данные не числовые
        try:
            values[key] = float(values[key])
        except:
            # Эта ф-и в отличии от предыдущей в случае ошибки возвращает None тк если ошибки нет то она должна будет вернуть
            # преобразованнеы в числовые значения словаря
            #print(key, values[key]) #FIXME Дебаг
            return None
    return values


def enter_window():

    # Текстовые поля и поляя ввода окна
    layout = [
        [sg.Text('Введите данные о проекте', font=("Helvetica", 18))],
        [sg.Text('')],
        [sg.Text('Название проекта'), sg.InputText('Открытие ателье по пошиву платьев', size=(49, 1))],
        [sg.Text('Автор проекта      '), sg.InputText('Зубкова М.А.', size=(49, 1))],
        [sg.Text('')],
        [sg.Cancel('Отмена'), sg.Submit('Продолжить')]
    ]

    # Создаем обьект Окно
    window = sg.Window('Форма для ввода информации о проекте', layout)

    #  Запускаем бесконченый цикл для отлавливания нажатия кнопок окна
    while True:

        # Считываем введенные значения в словарь values, ключами в словаре являются целые числа от 0
        # А нажатые кнопки в текстовую переменную event
        event, values = window.read()
        if event in (None, 'Отмена', sg.WIN_CLOSED):
            window.close()
            return None

        # Если нажата кнопка Продолжить
        if event == 'Продолжить':
            # Проверяем не пустые ли поля
            #print(values) #FIXME Дебаг
            if check_text_values(values):
                window.close()
                return values
            else:
                # Инчае показываем высплывающее окно ошибки
                sg.popup_error('Поля не должны быть пустыми!')


def data_window(class_levels):

    # Текстовые поля и поляя ввода окна
    layout = [
        [sg.Text('Введите инвестиционные данные', font=("Helvetica", 18))],
        [sg.Text('')],
        [sg.Text('Дата начала проекта'),
            sg.CalendarButton('Выбрать', target='start', format='%d.%m.%Y', default_date_m_d_y=(1, 1, 2020), ),
            sg.Input('01.01.2020', key='start', size=(25, 1))],
        [sg.Text('')],
        [sg.Text('Длительность', font=("Helvetica", 16))],
        [sg.Text('Лет        '), sg.InputText('5', key='y', size=(45, 1))],
        [sg.Text('Месяцев'), sg.InputText('0', key='m', size=(45, 1))],
        [sg.Text('')],
        [sg.Text('Первоначальные инвестиции'), sg.InputText('891000', key='i', size=(28, 1))],
        [sg.Text('Число периодов реализации проекта'), sg.InputText('5', key='n', size=(21, 1))],
        [sg.Text('Норма дисконта'), sg.InputText('0.112', key='r', size=(39, 1))],
        [sg.Text('Чистый поток платежей в периоде'), sg.InputText('671900', key='cf', size=(24, 1))],
        [sg.Text('Планируемая выручка'), sg.InputText('2970000', key='rev', size=(34, 1))],
        [sg.Text('')],
        [sg.Text('Пороговые значения', font=("Helvetica", 16))],
        [sg.Text('NPV'), sg.InputText(class_levels['NPV'], key='NPV', size=(10, 1))],
        [sg.Text('IRR '), sg.InputText(class_levels['IRR'], key='IRR', size=(10, 1))],
        [sg.Text('PP  '), sg.InputText(class_levels['PP'], key='PP', size=(10, 1))],
        [sg.Text('PI   '), sg.InputText(class_levels['PI'], key='PI', size=(10, 1))],
        [sg.Text('')],
        [sg.Cancel('Отмена'), sg.Submit('Рассчитать')]
    ]

    # Создаем обьект Окно
    window = sg.Window('Форма для ввода инвестиционных данных', layout)

    #  Запускаем бесконченый цикл для отлавливания нажатия кнопок окна
    while True:
        # В этом окне уже ключами словаря явлются текстовые поля заданные атрибутами формы key
        event, values = window.read()

        # Тк window.read() возвращает значения всех элементов формы окна, в тч интерактивной кнопки 'Выбрать'
        # удаляем это значение из словаря как мусорное (сама дата будет возвращена с ключем следующего после кнопки поля)
        values.pop('Выбрать')

        #print(values) #FIXME Дебаг
        if event in (None, 'Отмена', sg.WIN_CLOSED):
            window.close()
            return None
        if event == 'Рассчитать':
            # Сначала проверяем не пропущены(не пустые) ли какие-то поля
            if not check_text_values(values):
                sg.popup_error('Поля не должны быть пустыми!')
                # Запускаем новую итерацию цикла проверки введеных значений
                continue

            # Далее проверяем и одновремнно конвертируем поля в числовой тип
            values = check_data_values(values)
            if values is None:
                sg.popup_error('Поля должны быть числовыми!')
                # Запускаем новую итерацию цикла проверки введеных значений
                continue

            # Если проверки прошли то цикл доходит до этого места где мы зхакрываем окно и передаем ковертируемые значения
            window.close()
            return values


def result_window(result_dict, enter_values, data_values):
    # Генерируем вектор Х - года реализации проекта, начиная с года даты, выбранной в форме, по кол-ву периодов проекта
    # Разбиваем дату на список из 3-х состовляющих даты
    date_list = data_values['start'].split('.')
    # Год - это последний элемент этого списка, сразу приводдим его к целочисленному типу
    start_year = int(date_list[len(date_list) - 1])
    # Число лет на графике, включая год сарта, будет равно кол-ву периодов + 1
    num_years = int(data_values['n']) + 1
    x = [start_year + i for i in range (num_years)]

    #y2 = [-891, -286.3, 257.95, 758.95, 1214.92, 1637.36]
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = 7, 16
    plt.rcParams.update({'font.size': 7})

    # Описываем верзний раздел окна результатов
    plt.subplot(3, 1, 1)
    plt.grid(False)
    plt.axis(False)

    text = f'Проект: {enter_values[0]}\nАвтор: {enter_values[1]}\nДата начала проекта: {data_values["start"]}\n' \
           f'Инвестиции: {data_values["i"]}\n\n'
    # Получаем значения из словаря результатов, округляя для удобного представления
    NPV = round(result_dict['npv'])
    PI = round(result_dict['pi'], 2)
    IRR = round(result_dict['irr'], 2)
    PP = round(result_dict['pp'], 2)
    text += f'NPV = {NPV}\nIRR = {IRR}\nPP = {PP}\nPI = {PI}\n\n'

    # Данные для выода дискриминатной функции
    b1 = round(get_koef(data_values['NPV']), 8)
    b2 = round(get_koef(data_values['IRR']), 4)
    b3 = round(get_koef(data_values['PP']), 4)
    b4 = round(get_koef(data_values['PI']), 4)

    # Расчет дискриминатной функции
    x_dict = dict()
    x_dict['NPV'] = NPV
    x_dict['IRR'] = IRR
    x_dict['PP'] = PP
    x_dict['PI'] = PI
    D = round(discriminant_function(data_values, x_dict), 1)
    text += f'Дискриминатная функция D\n{b1} * {NPV} + {b2} * {IRR} - {b3} * {PP} + {b4} * {PI} - 2 = {D}\n\n'

    # Результат дискриминатного анлаиза по отнесенному классу
    RES = discriminant_analysis(data_values, x_dict)
    text += f'Класс проекта: {RES}'

    plt.figtext(0.05, 0.7, text, fontsize=12, horizontalalignment='left')

    # Средний раздел окна результатов - график PV
    # Значения y графика получаем из функции генерирования PV списка
    y1 = generate_pv_row(data_values)
    plt.subplot(3, 1, 2)
    plt.bar(x, y1, color = 'indigo')
    #plt.title("График PV", fontsize=10)
    for i in range(len(x)):
        y_pos = y1[i]
        if y_pos < 0:
            y_pos = 0
        plt.annotate(str(y1[i]), xy=(x[i], y_pos))
    plt.ylabel('PV (т.руб)', fontsize=8, color='indigo')
    plt.grid(True)

    # Нижний раздел окна результатов - график NPV
    # Фнукция generate_npv_row() кроме массива значение Npv Для гарфика возращает точку dpp
    y2, dpp, dpp_year = generate_npv_row(data_values, x)
    plt.subplot(3, 1, 3)
    plt.plot(x, y2, color = 'indigo', marker='.')
    for i in range(len(x)):
        plt.annotate(str(y2[i]), xy=(x[i] + 0.2, y2[i]))
    # Отмечаем точку DPP
    plt.plot(dpp, 0, 'd', color = 'indigo')
    plt.annotate(f'DPP={dpp_year}', xy=(dpp + 0.2, 0), weight='heavy')

    plt.ylabel('NPV (т.руб)', fontsize=8, color='indigo')
    plt.grid(True)
    plt.show()


def get_result(enter_values, data_values):
    # Возвращаем значения в форме словаря
    res = dict()
    res['npv'] = count_npv(data_values)
    res['pi'] = count_pi(data_values, res['npv'])
    res['irr'] = count_irr(data_values)
    res['pp'] = count_pp(data_values)
    return res


# Внимание! тк в форме ввода инвестиционных данных для чистого потока платежей по заданию отвдится только одно поле
# считаем cash flow постоянным для всех периодов
def count_npv(data_values, n=-1):
    # Второй необязательный аргумент - число преиодов, если он не передается в функцию то берем его из данных формы
    if n == -1:
        # Все значения конвертированы в Float, а число периодов должно быть целое число
        n = int(data_values['n'])
    # Ставка дисконтирования
    r = data_values['r']
    # Чистый поток
    cf = data_values['cf']

    # Генерируем список потоков на N периодов
    cf_list = []
    # В качесте нулевого потока добавляем отрицательные инвестиции
    cf_list.append(-data_values['i'])
    # Добавляем чистый поток по числу периодов
    for _ in range(n):
        cf_list.append(cf)
    # Используем функцию библиотеки Numpy
    npv = npf.npv(r, cf_list)
    return npv


def count_irr(data_values):
    # Число периодов
    # Все значения конвертированы в Float, а число периодов должно быть целое число
    n = int(data_values['n'])
    # Чистый поток
    cf = data_values['cf']

    # Генерируем список потоков на N периодов
    cf_list = []
    # В качесте нулевого потока добавляем отрицательные инвестиции
    cf_list.append(-data_values['i'])
    # Добавляем чистый поток по числу периодов
    for _ in range(n):
        cf_list.append(cf)
    # Используем функцию библиотеки Numpy
    irr = npf.irr(cf_list)
    return irr


def count_pp(data_values):
    # Чистый поток
    cf = data_values['cf']
    # Инвестиции
    inv = data_values['i']
    return inv / cf


def count_pi(data_values, npv):
    # Инвестиции
    inv = data_values['i']
    # Тк npv уже расчитана впосользуемся ей для вычисления PI
    pi = (npv + inv) / inv
    return pi


def generate_pv_row(data_values):
    cf_list = []
    # Конвертируем все значения из руб в тыс руб
    # Первое значение приведенного денежного потока - это инвестиции
    cf_start = -round(data_values['i'] / 1000, 2)
    cf_list.append(cf_start)
    # Число периодов, включая стартовый год
    n = int(data_values['n']) + 1
    for i in range(1, n):
        cf = data_values['cf'] / 1000
        pv = round(cf / ((1 + data_values['r']) ** i), 2)
        cf_list.append(pv)
    return cf_list


def generate_npv_row(data_values, x):
    npv_list = []
    # Число периодов, включая стартовый год
    n = int(data_values['n']) + 1
    for i in range(n):
        npv = round(count_npv(data_values, i) / 1000, 2)
        npv_list.append(npv)

    # Находим DPP
    for i in range(len(npv_list)):
        # Нахходим точку перехода дисконтированного потока в положительную зону
        if i > 0 and npv_list[i] > 0 and npv_list[i - 1] <= 0:
            start = npv_list[i - 1]
            end = npv_list[i]
            # период в котором происходит окупаемость
            x0 = i - 1
            break
    # Вычисляем точку dpp геометрически как место пересечения граифка оси х
    delta = abs(start) / (end - start)
    dpp = x[x0] + delta
    year = round(delta + x0, 2)
    return npv_list, dpp, year


def discriminant_analysis(class_levels, x_dict):
    # Тк дискриминатная функция уже реализована, просто вызываем ее
    d = discriminant_function(class_levels, x_dict)
    if d < 0:
        return "Не привлекателен"
    return "Привлекателен"


def main():
    # Получаем данные из первого окна
    enter_values = enter_window()
    # Если нажата кнопка Отмена
    if not enter_values:
        return 0
    #print(enter_values) #FIXME Дебаг

    # Получаем пороговые значения NPV, IRR, PP, PI
    class_levels = get_class_levels()

    # Получаем валидированные данные второго окна
    data_values = data_window(class_levels)
    # Если нажата кнопка Отмена
    if not data_values:
        return 0
    #print(data_values) #FIXME Дебаг

    # Функция прослойка вызывающая функции вычисления всех значений
    result_dict = get_result(enter_values, data_values)
    result_window(result_dict, enter_values, data_values)


if __name__ == "__main__":
    main()
