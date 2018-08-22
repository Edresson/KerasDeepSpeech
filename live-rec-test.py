## this file is created to test WER/LER against live audio
## it should be run on a desktop/laptop using a microphone to speak
## either you can type in the transcript yourself or let audio transcribe for the GroundTruth label

#using silence is too complicated. Just use ctrl+c as works on all systems and can cope with background noise

import argparse
import datetime
import wave
from os import path

import sys
import GUI as gui
from PyQt5 import QtCore, QtGui, QtWidgets

import pandas as pd
import pyaudio
import speech_recognition as sr
from model import *
from report import *
from utils import *

from data import combine_all_wavs_and_trans_from_csvs
from generator import *
import threading
#####################################################

def threaded(func):
    def wrapper(*_args, **kwargs):
        t = threading.Thread(target=func, args=_args)
        t.start()
        return
    return wrapper

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "./live/rec.wav")
OUTPUT_DIR = "./live/"

flag = 0

def clean(word):
    # token = re.compile("[\w-]+|'m|'t|'ll|'ve|'d|'s|\'")
    ## LC ALL & strip fullstop, comma and semi-colon which are not required
    new = word.lower().replace('.', '')
    new = new.replace(',', '')
    new = new.replace(';', '')
    new = new.replace('"', '')
    new = new.replace('!', '')
    new = new.replace('?', '')
    new = new.replace(':', '')
    new = new.replace('-', '')
    return new

@threaded 
def startloop(rec_number=1):
    K.set_learning_phase(0)
    global ui,flag,y_pred,input_data,K,pd,sr,OUTPUT_DIR,model_input_type
    if flag == 0:
        ui.bt_gravar.setText('Parar')
        flag= 1
        ##read in data from csv
        # df = pd.read_csv(TRANSCRIPT_SOURCE, sep=',', header=None)

        #HEADERS
        wav_filename = []
        wav_filesize = []
        transcript = []

        # print("when ready press enter to start recording and then ctrl+c to stop")
        # time.sleep(1)

        trans = ""#str(raw_input('please type the exact words you will speak (for WER calculation), or press enter to use Google Transcribe for WER calc\n:'))
        trans = clean(trans)
        if trans == "":
            trans = "N/A"

        print("Transcript is:", trans)

        inputvar = ''#str(raw_input('ready? press enter to begin recording and ctrl+c to stop'))
        filename = "rec"

        if inputvar == "":
            r = record(filename, OUTPUT_DIR, trans)
            # inputcheck = str(raw_input('press enter if you are happy, or r to redo.'))
            wav_filename.append(r)
            wav_filesize.append(os.path.getsize(r))

            if trans == "N/A":
                r = sr.Recognizer()
                with sr.AudioFile(AUDIO_FILE) as source:
                    audio = r.record(source)  # read the entire audio file
                    trans = r.recognize_google(audio)
                    trans = trans.lower()
            print 'transcript: '+trans
            transcript.append(trans)


        a = {'wav_filename': wav_filename,
             'wav_filesize': wav_filesize,
             'transcript': transcript
             }

        df_train = pd.DataFrame(a, columns=['wav_filename', 'wav_filesize', 'transcript'], dtype=int)
        df_train.to_csv("live/live.csv", sep=',', header=True, index=False, encoding='ascii')
        args.test_files = "live/live.csv"
        print("Getting data from arguments")
        test_dataprops, df_test = combine_all_wavs_and_trans_from_csvs(args.test_files, sortagrad=False)
        print test_dataprops, df_test 
        ## x. init data generators
        print("Creating data batch generators")
        testdata = BatchGenerator(dataframe=df_test, dataproperties=test_dataprops,
                                      training=False, batch_size=1, model_input_type=model_input_type)

        print("testedata",testdata)
        ## RUN TEST
        report = K.function([input_data, K.learning_phase()], [y_pred])
        print ("report",report)
        report_cb = ReportCallback(report, testdata, model, args.name, save=False)
        report_cb.force_output = True
        label,corrected=report_cb.live_predict(0, logs=None)
        ui.plainTextEdit.setPlainText(str(label+"=="+corrected))
    else:
        flag = 0        
        ui.bt_gravar.setText('Gravar')

def record(name, dir, trans):
    global flag
    global ui,flag,y_pred,input_data,K,pd,sr,OUTPUT_DIR
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK)

    #print("Recording to:", dir, name)
    print ("\n" * 30) #works on all OS
    print("RECORDING, press ctrl+c to stop recording")
    print(trans)
    frames = []
    #for i in range(0, int(RATE / CHUNK * RECORD_MINUTES * 60)):

    while flag:
            data = stream.read(CHUNK)
            frames.append(data)


    stream.stop_stream()
    stream.close()
    p.terminate()

    print("Finished recording to:", dir, name)

    fileindir = dir+name+".wav"

    wf = wave.open(fileindir, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return fileindir


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #parser.add_argument('--loadcheckpointpath', type=str, default='./checkpoints/trimmed/',
    #parser.add_argument('--loadcheckpointpath', type=str, default='./checkpoints/epoch/LER-WER-best-DS3_2017-09-02_13-40',
    parser.add_argument('--loadcheckpointpath', type=str, default='checkpoints/epoch/LER-WER-best-DS8_2018-07-14_15-07',
                        help='If value set, load the checkpoint json '
                             'weights assumed as same name '
                             ' e.g. --loadcheckpointpath ./checkpoints/'
                             'TRIMMED_ds_ctc_model ')
    parser.add_argument('--model_arch', type=int, default=8,
                        help='choose between model_arch versions (when training not loading) '
                             '--model_arch=1 uses DS1 fully connected layers with simplernn'
                             '--model_arch=2 uses DS2 fully connected with GRU'
                             '--model_arch=3 is custom model')
    parser.add_argument('--name', type=str, default='',
                        help='name of run')


    args = parser.parse_args()
    runtime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    if args.name == "":
        args.name = "DS" + str(args.model_arch) + "_" + runtime


    # check any special data model requirments e.g. a spectrogram
    print(args)
    if(args.model_arch == 1):
        model_input_type = "mfcc"
    elif(args.model_arch == 2 or args.model_arch == 5):
        print("Spectrogram required")
        # spectrogram = True
        model_input_type = "spectrogram"
    else:
        model_input_type = "mfcc"


    #1. load model
    if args.loadcheckpointpath:
        # load existing
        print("Loading model")

        cp = args.loadcheckpointpath
        assert(os.path.isdir(cp))
        trimmed = False

        if trimmed:
            model_path = os.path.join(cp, "TRIMMED_ds_model")
        else:
            model_path = os.path.join(cp, "model")
        # assert(os.path.isfile(model_path))

        model = load_model_checkpoint(model_path)
        opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

        print("Model loaded")

    else:
        # new model
        raise("You need to load an existing trained model")



    model.compile(optimizer=opt, loss=ctc)

    try:
        y_pred = model.get_layer('ctc').input[0]
    except Exception as e:
        print("error", e)
        print("couldn't find ctc layer, possibly a trimmed layer, trying other name")
        y_pred = model.get_layer('out').output

    input_data = model.get_layer('the_input').input
    K.set_learning_phase(0)


    #2. record data and put it in live folder LOOP

    
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = gui.Ui_MainWindow()
    ui.setupUi(MainWindow)


    ui.bt_gravar.clicked.connect(startloop)


    MainWindow.show()
    sys.exit(app.exec_())


    K.clear_session()
