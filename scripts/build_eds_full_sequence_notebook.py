#!/usr/bin/env python
"""Build a separate full-sequence EDS training notebook; leave the clip notebook intact."""
import argparse
import copy
import json
from pathlib import Path
import runpy
import textwrap

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'notebooks'/'02_eds_full_sequence_4dgs_training.ipynb'


def source(text):
    return (textwrap.dedent(text).strip()+'\n').splitlines(keepends=True)


def build():
    base = runpy.run_path(str(ROOT/'scripts'/'build_eds_notebook.py'))
    cells = copy.deepcopy(base['CELLS'])
    cells[0]['source'] = source('''
    # آموزش 4DGS و ارزیابی TIDES روی کل سکانس EDS

    این نوت‌بوک جدا برای کل `00_peanuts_dark` است، نه کلیپ ۱٫۵ثانیه‌ای.
    فایل را در پوشه `notebooks` مخزن `TIDES_2026` روی سرور بگذارید، کرنل **Python (tides)**
    را انتخاب کنید و **Run All** بزنید. سلول اول کد شامل مسیرهای شماست؛ نیازی به افزودن سلول نیست.

    **Run All آماده‌سازی و سپس آموزش واقعی GPU را شروع می‌کند**؛ پس از آن شبیه‌سازی و ارزیابی
    دو حالت TIDES اجرا می‌شوند. baselineها و ablationها در این اجرا انتخاب نشده‌اند.
    خروجی جداگانه در `work/eds/00_peanuts_dark_full` ذخیره می‌شود.
    اگر صحنه کامل از قبل آماده باشد دوباره ساخته نمی‌شود؛ چک‌پوینت کامل نیز دوباره آموزش نمی‌بیند.
    خروجی ناقص خودکار حذف یا بازنویسی نمی‌شود.

    همه فریم‌های داخل بازه پوز معتبر، با stride=1 انتخاب می‌شوند؛ محدودیت ۱٫۵ ثانیه و سقف ۶۰۰
    فریم وجود ندارد. تفکیک قبلی آموزش/ارزیابی و تنظیمات ۳۰۰۰ coarse + ۱۴۰۰۰ fine حفظ شده‌اند.
    نتیجه هنوز مربوط به یک سکانس است و معادل میانگین کل EDS مقاله نیست.
    قرارداد GT=RGB و وضعیت flip همان فرض‌های ثبت‌شده قبلی هستند؛ تغییر خودکاری در آن‌ها نمی‌دهیم.
    ''')
    first = dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=source('''
    # تنظیمات از قبل برای سرور شما؛ اجرای سلول‌های اصلی نیاز به export ترمینال ندارد.
    import os
    os.environ['TIDES_EDS_ROOT'] = '/u/marmary/evtween/EDS/EDS'
    os.environ['TIDES_EDS_CALIB'] = '/u/marmary/evtween/EDS/calib/00_calib/camchain-mediajaviJAVISdatasetshwdscalibratione2kalibr.yaml'
    os.environ['TIDES_SEQUENCE'] = '00_peanuts_dark'
    os.environ['TIDES_WORK'] = '/u/marmary/airlab_v2e/TIDES_2026/work/eds/00_peanuts_dark_full'
    os.environ['TIDES_FOURDGS'] = '/u/marmary/airlab_v2e/TIDES/third_party/4DGaussians'
    os.environ['TIDES_GT_POSE_FRAME'] = 'rgb'
    os.environ['TIDES_TANGENT_MODE'] = 'reverse'
    os.environ['TIDES_SEED'] = '0'
    os.environ['TIDES_RUN_STAGES'] = 'prepare,audit,train,tides,evaluate,record'
    # A previous clip session must not redirect full-sequence metrics or bounds.
    os.environ.pop('TIDES_T_START', None)
    os.environ.pop('TIDES_T_END', None)
    os.environ.pop('TIDES_METRICS', None)
    print('Full-sequence workflow selected. Training/simulation start when their cells run.')
    '''))
    cells.insert(1,first)
    # Identify inherited cells by content rather than positions after insertion.
    for c in cells:
        text = ''.join(c['source'])
        if c['cell_type'] == 'code' and 'EDS_ROOT = Path(' in text:
            text = text.replace("T_START, T_END = float(os.environ.get('TIDES_T_START', '0')), float(os.environ.get('TIDES_T_END', '1.5'))", 'T_START, T_END = 0.0, None  # actual supported endpoints are read after preparation')
            text = text.replace("or not T_START < T_END:", 'or (T_END is not None and not T_START < T_END):')
            c['source'] = source(text)
        if c['cell_type']=='markdown' and '## 1. Original EDS paths' in text:
            c['source'] = source('''
            ## 1. مسیرهای آماده و پیش‌نیازها

            مسیر checkout قبلی 4DGaussians و حالت reverse از قبل تنظیم شده‌اند.
            هیچ وابستگی‌ای خودکار نصب نمی‌شود. صحنه کوتاه قبلی در پوشه دیگری باقی می‌ماند.
            این نوت‌بوک به همان اسکریپت‌های اصلاح‌شده مخزن نیاز دارد (کامیت 8f612d9 یا نسخه بعد از آن).
            ''')
        if c['cell_type']=='markdown' and '## 2. Inspect, prepare' in text:
            c['source'] = source('''
            ## 2. آماده‌سازی کل سکانس و کنترل پوشش فریم‌ها

            آماده‌سازی بدون t-end و با سقف sys.maxsize اجرا می‌شود. بعد از آماده‌سازی، بازه زمانی
            از tides_meta.json خوانده می‌شود؛ همان کاری که سلول تنظیمات دستی قرار بود انجام دهد،
            اما اینجا پس از ساخته‌شدن فایل اجرا می‌شود. تعداد و زمان فریم‌ها با کل بازه معتبر دیتاست
            مقایسه می‌شود تا صحنه کوتاه یا آماده‌سازی ناقص به‌اشتباه به عنوان اجرای کامل استفاده نشود.
            ''')
        if c['cell_type']=='code' and 'def check_scene():' in text:
            prefix = '''
            # Check prerequisites before creating a new scene or starting training.
            import torch
            for required in [EDS_ROOT/SEQUENCE, EDS_CALIB, FOURDGS/'train.py',
                             FOURDGS/'scene'/'dataset_readers.py', CFG4D,
                             ROOT/'scripts'/'audit_prepared_scene.py', ROOT/'scripts'/'compare_eds_paper.py']:
                if not required.exists():
                    raise FileNotFoundError(f'Missing required path: {required}')
            if not torch.cuda.is_available():
                raise RuntimeError('Select the tides CUDA kernel before running the full workflow')
            print('GPU:', torch.cuda.get_device_name(0))

            def prepare_full_sequence():
                meta_path = WORK/'tides_meta.json'
                if meta_path.is_file():
                    print('Reusing prepared scene; validating full coverage below:', WORK)
                elif 'prepare' in STAGES:
                    if (WORK/'scene').exists():
                        raise FileExistsError('Incomplete scene without metadata; use a fresh WORK, do not overwrite it')
                    run(sys.executable, ROOT/'scripts'/'prepare_scene.py', '--dataset','eds','--root',EDS_ROOT,
                        '--seq',SEQUENCE,'--out',WORK,'--calib',EDS_CALIB,'--gt-pose-frame',GT_POSE_FRAME,
                        '--t-start',0,'--stride',1,'--max-frames',sys.maxsize,'--seed',SEED)
                else:
                    raise FileNotFoundError('No prepared full scene. Enable prepare or select its WORK.')
                return json.loads(meta_path.read_text())

            prepared = prepare_full_sequence()
            T_START, T_END = map(float, prepared['t_range'])
            os.environ['TIDES_T_START'], os.environ['TIDES_T_END'] = str(T_START), str(T_END)
            print('Full-scene selected frames:', len(prepared['frame_times_s']))
            print('Actual supported time range:', [T_START,T_END])

            # Loading sequence metadata does not load the full event stream.
            import numpy as np
            from tides.datasets import load_sequence
            original = load_sequence('eds', str(EDS_ROOT), SEQUENCE,
                                     calib_path=str(EDS_CALIB), gt_pose_frame=GT_POSE_FRAME)
            if original.gt_poses is None:
                raise ValueError('Original EDS GT poses are required')
            times = original.frame_times
            lo, hi = original.gt_poses[0][0], original.gt_poses[0][-1]
            expected = times[(times >= 0) & (times >= lo) & (times <= hi)]
            actual = np.asarray(prepared['frame_times_s'])
            if len(expected) != len(actual) or not np.allclose(expected, actual, rtol=0, atol=1e-7):
                raise ValueError('Existing WORK does not cover every frame in the valid GT interval. Use a fresh full WORK.')
            print('Source RGB frames:', len(times), '| selected:', len(actual), '| outside valid GT:', len(times)-len(actual))
            '''
            # Reuse original root/sequence/pose/calibration/upstream checks after full preparation.
            inherited = text[text.index('def check_scene():'):]
            c['source'] = source(prefix) + ['\n'] + source(inherited)
        if c['cell_type']=='markdown' and '## 5. Results' in text:
            c['source'] = source(text.replace('Single-clip', 'Single-sequence'))
        if c['cell_type']=='code' and "PAPER REPRODUCTION: NOT ESTABLISHED." in text:
            text = text.replace('This clip is compared with EDS dataset means.', 'This full sequence is compared with EDS dataset means.')
            text = text.replace("'requested_range':[T_START,T_END],", "'requested_range':[T_START,T_END],'coverage':'all RGB frames within valid GT range',")
            text = text.replace("comparison[comparison['table'] == table].drop(columns='table').reset_index(drop=True)", "comparison[comparison['table'] == table].drop(columns='table').reset_index(drop=True).fillna('—')")
            c['source'] = source(text)
    for i,c in enumerate(cells):
        c['id'] = f'eds-full-{i:02d}'
    notebook = dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python (tides)',language='python',name='tides'),
                    language_info=dict(name='python')), nbformat=4,nbformat_minor=5)
    return json.dumps(notebook, ensure_ascii=False, indent=1)+'\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=TARGET)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build(),encoding='utf-8')
    print(args.out)


if __name__=='__main__':
    main()
